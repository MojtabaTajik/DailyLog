"""RAG sidecar for the DailyLog bot.

Stores chunk embeddings of notes in LanceDB. Embeddings come from a
local Ollama instance running nomic-embed-text. Chunks are keyed by an
opaque string `key` chosen by the caller (typically the file's path
relative to the vault root). Reindexing a key deletes all prior chunks
for that key and inserts the new ones.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import lancedb
import pyarrow as pa
from fastapi import FastAPI, HTTPException
from lancedb.rerankers import RRFReranker
from pydantic import BaseModel, Field

LANCEDB_PATH = os.environ.get("LANCEDB_PATH", "/data/lancedb")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
TABLE_NAME = "notes"
EMBED_DIM = 768  # nomic-embed-text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")

state: dict[str, Any] = {}


def chunk_markdown(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split markdown into roughly paragraph-shaped chunks."""
    text = text.strip()
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        if overlap and chunks:
            tail = chunks[-1][-overlap:]
            buf = f"{tail}\n\n{p}"
        else:
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


async def embed_batch(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    """Embed all texts concurrently. Ollama serializes them internally
    on CPU, but firing requests in parallel still avoids the per-call
    HTTP round-trip cost piling up sequentially.
    """
    async def _one(t: str) -> list[float]:
        r = await client.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["embedding"]

    return list(await asyncio.gather(*(_one(t) for t in texts)))


REQUIRED_FIELDS = {"text", "vector", "key", "chunk_id", "content_hash"}


def _build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
            pa.field("key", pa.string()),
            pa.field("chunk_id", pa.int32()),
            pa.field("content_hash", pa.string()),
        ]
    )


def open_table(db: lancedb.DBConnection):
    if TABLE_NAME in db.table_names():
        tbl = db.open_table(TABLE_NAME)
        existing = {f.name for f in tbl.schema}
        if REQUIRED_FIELDS.issubset(existing):
            return tbl
        # Schema mismatch (e.g. older `date`-keyed layout). Drop and
        # recreate; the index can be rebuilt with the bot's /reindex.
        log.warning("table schema is outdated (%s); dropping and recreating", existing)
        db.drop_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=_build_schema())


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _quote(s: str) -> str:
    """SQL-quote a string for use in a LanceDB where clause."""
    return s.replace("'", "''")


def existing_hash_for(table, key: str) -> str | None:
    """Return the content_hash currently stored for key, or None."""
    rows = (
        table.search()
        .where(f"key = '{_quote(key)}'")
        .select(["content_hash"])
        .limit(1)
        .to_list()
    )
    if not rows:
        return None
    return rows[0].get("content_hash")


def ensure_fts_index(table) -> None:
    """Build (or rebuild) the BM25 full-text index on the text column.

    Hybrid search needs this so exact-token queries (proper nouns,
    rare words) do not get washed out by dense embeddings.
    """
    try:
        table.create_fts_index("text", replace=True)
    except Exception as e:  # noqa: BLE001 - sidecar should not crash on this
        log.warning("create_fts_index failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["db"] = lancedb.connect(LANCEDB_PATH)
    state["table"] = open_table(state["db"])
    state["http"] = httpx.AsyncClient()
    state["reranker"] = RRFReranker()
    if state["table"].count_rows() > 0:
        ensure_fts_index(state["table"])
    log.info("rag service ready: lancedb=%s ollama=%s", LANCEDB_PATH, OLLAMA_HOST)
    try:
        yield
    finally:
        await state["http"].aclose()


app = FastAPI(lifespan=lifespan)


class IndexRequest(BaseModel):
    key: str = Field(..., description="Opaque identifier, e.g. note path relative to vault root")
    content: str
    rebuild_fts: bool = Field(
        True,
        description="Rebuild the BM25 full-text index after this insert. Set to false during bulk reindex and call /rebuild-fts once at the end.",
    )


class QueryRequest(BaseModel):
    q: str
    k: int = 6


class Hit(BaseModel):
    text: str
    key: str
    chunk_id: int
    score: float


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index")
async def index(req: IndexRequest) -> dict[str, Any]:
    table = state["table"]
    http: httpx.AsyncClient = state["http"]

    new_hash = sha256_hex(req.content)

    # Skip if already indexed with the same hash. This makes /reindex
    # cheap to re-run across the whole vault.
    if existing_hash_for(table, req.key) == new_hash:
        return {"chunks": 0, "skipped": True, "hash": new_hash}

    table.delete(f"key = '{_quote(req.key)}'")

    chunks = chunk_markdown(req.content)
    if not chunks:
        return {"chunks": 0, "skipped": False, "hash": new_hash}

    vectors = await embed_batch(http, chunks)
    rows = [
        {
            "text": c,
            "vector": v,
            "key": req.key,
            "chunk_id": i,
            "content_hash": new_hash,
        }
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    table.add(rows)
    if req.rebuild_fts:
        ensure_fts_index(table)
    log.info("indexed %d chunks for %s (fts_rebuilt=%s)", len(rows), req.key, req.rebuild_fts)
    return {"chunks": len(rows), "skipped": False, "hash": new_hash}


@app.post("/rebuild-fts")
async def rebuild_fts() -> dict[str, str]:
    """Rebuild the BM25 full-text index. Call this once at the end of
    a bulk reindex when /index was invoked with rebuild_fts=false.
    """
    ensure_fts_index(state["table"])
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict[str, dict[str, str]]:
    """Return a map of indexed key -> content_hash."""
    table = state["table"]
    rows = table.search().select(["key", "content_hash"]).limit(1_000_000).to_list()
    by_key: dict[str, str] = {}
    for r in rows:
        k = r["key"]
        h = r.get("content_hash") or ""
        # Same hash repeats across a key's chunks; first wins.
        by_key.setdefault(k, h)
    return {"keys": by_key}


@app.post("/query")
async def query(req: QueryRequest) -> dict[str, list[Hit]]:
    if not req.q.strip():
        raise HTTPException(400, "empty query")

    table = state["table"]
    http: httpx.AsyncClient = state["http"]

    [qvec] = await embed_batch(http, [req.q])

    # Hybrid: BM25 catches exact tokens (proper nouns, rare words);
    # vector catches semantic matches. RRF fuses the two rankings.
    try:
        results = (
            table.search(query_type="hybrid", vector_column_name="vector")
            .vector(qvec)
            .text(req.q)
            .rerank(reranker=state["reranker"])
            .limit(req.k)
            .to_list()
        )
    except Exception as e:
        # Most likely cause: FTS index not yet built. Fall back to
        # dense-only and rebuild for next time.
        log.warning("hybrid search failed (%s), falling back to vector-only", e)
        ensure_fts_index(table)
        results = table.search(qvec).limit(req.k).to_list()

    hits = [
        Hit(
            text=r["text"],
            key=r["key"],
            chunk_id=int(r["chunk_id"]),
            score=float(r.get("_relevance_score", r.get("_distance", 0.0))),
        )
        for r in results
    ]
    return {"hits": hits}
