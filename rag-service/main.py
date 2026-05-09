"""RAG sidecar for the DailyLog bot.

Stores chunk embeddings of notes in LanceDB. Embeddings come from a
local Ollama instance running nomic-embed-text. Chunks are keyed by an
opaque string `key` chosen by the caller (typically the file's path
relative to the vault root). Each chunk also stores the parsed date,
weekday, and Markdown section (## heading) it came from, so retrieval
can filter by date range and the LLM gets temporal context for free.
"""

import asyncio
import datetime as dt
import hashlib
import os
import re
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
import lancedb
import pyarrow as pa
from fastapi import FastAPI, HTTPException
from lancedb.rerankers import RRFReranker
from pydantic import BaseModel, Field

LANCEDB_PATH = os.environ.get("LANCEDB_PATH", "/data/lancedb")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.0"))
TABLE_NAME = "notes"
EMBED_DIM = 768  # nomic-embed-text
MAX_PER_KEY = 2  # diversity cap per source note

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")

state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_DAILY_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.md$")


def parse_date_from_key(key: str) -> tuple[str, str]:
    """Extract (YYYY-MM-DD, weekday) from a daily-note key, ("","") otherwise.

    Daily notes live at `Daily notes/YYYY/MM/YYYY-MM-DD.md`. Other vault
    files just get empty strings and are searchable by content alone.
    """
    m = _DAILY_RE.search(key)
    if not m:
        return "", ""
    iso = m.group(1)
    try:
        d = dt.date.fromisoformat(iso)
    except ValueError:
        return "", ""
    return iso, d.strftime("%a")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_H2_RE = re.compile(r"^##\s+(.+?)\s*$")


def chunk_markdown(text: str, size: int = 800, overlap: int = 120) -> list[tuple[str, str]]:
    """Split markdown into (section, body) chunks.

    Sections are H2 headings (`## Foo`). A chunk never crosses a section
    boundary so retrieval can attribute each chunk to a specific
    section. For files with no H2 headings, section is "" and the
    behaviour matches the previous paragraph-based chunker.
    """
    sections: list[tuple[str, str]] = []
    current_name = ""
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_name, body))

    for line in text.splitlines():
        m = _H2_RE.match(line)
        if m:
            flush()
            current_name = m.group(1).strip()
            current_lines = []
            continue
        current_lines.append(line)
    flush()

    chunks: list[tuple[str, str]] = []
    for name, body in sections:
        for c in _pack(body, size, overlap):
            chunks.append((name, c))
    return chunks


def _pack(text: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs into ~size-char chunks with optional overlap."""
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


# ---------------------------------------------------------------------------
# Embedding text
# ---------------------------------------------------------------------------

def build_chunk_text(date: str, weekday: str, section: str, body: str) -> str:
    """Prepend Date/Section headers to a chunk body. Empty fields are
    omitted so non-daily files do not get a stub `Date:` line."""
    parts: list[str] = []
    if date:
        parts.append(f"Date: {date} ({weekday})" if weekday else f"Date: {date}")
    if section:
        parts.append(f"Section: {section}")
    if not parts:
        return body
    parts.append("---")
    parts.append(body)
    return "\n".join(parts)


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


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "text", "vector", "key", "chunk_id", "content_hash",
    "date", "weekday", "section",
}


def _build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
            pa.field("key", pa.string()),
            pa.field("chunk_id", pa.int32()),
            pa.field("content_hash", pa.string()),
            pa.field("date", pa.string()),
            pa.field("weekday", pa.string()),
            pa.field("section", pa.string()),
        ]
    )


def open_table(db: lancedb.DBConnection):
    if TABLE_NAME in db.table_names():
        tbl = db.open_table(TABLE_NAME)
        existing = {f.name for f in tbl.schema}
        if REQUIRED_FIELDS.issubset(existing):
            return tbl
        # Schema mismatch (e.g. older layout without date/section). Drop
        # and recreate; the bot's /reindex rebuilds everything.
        log.warning("table schema is outdated (%s); dropping and recreating", existing)
        db.drop_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=_build_schema())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _quote(s: str) -> str:
    """SQL-quote a string for use in a LanceDB where clause."""
    return s.replace("'", "''")


def _validate_key(key: str) -> None:
    if not key:
        raise HTTPException(400, "empty key")
    if "\n" in key or "\r" in key or "\x00" in key:
        raise HTTPException(400, "invalid key: control characters")


def existing_hash_for(table, key: str) -> Optional[str]:
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


# ---------------------------------------------------------------------------
# Temporal parser
# ---------------------------------------------------------------------------

_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_YM_RE = re.compile(r"\b(\d{4}-\d{2})\b(?!-)")
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def parse_temporal_range(q: str, today: Optional[dt.date] = None) -> Optional[tuple[str, str]]:
    """Best-effort extraction of an ISO date range from a question.

    Returns (date_from, date_to) inclusive, both YYYY-MM-DD, or None.
    Pure regex/string match. Week starts Monday.
    """
    if not q:
        return None
    today = today or dt.date.today()
    s = q.lower()

    m = _ISO_RE.search(s)
    if m:
        d = m.group(1)
        return d, d

    m = _YM_RE.search(s)
    if m:
        ym = m.group(1)
        try:
            first = dt.date.fromisoformat(ym + "-01")
        except ValueError:
            first = None
        if first:
            if first.month == 12:
                nxt = dt.date(first.year + 1, 1, 1)
            else:
                nxt = dt.date(first.year, first.month + 1, 1)
            last = nxt - dt.timedelta(days=1)
            return first.isoformat(), last.isoformat()

    if "today" in s:
        return today.isoformat(), today.isoformat()
    if "yesterday" in s:
        d = today - dt.timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if "this week" in s:
        start = today - dt.timedelta(days=today.weekday())
        end = start + dt.timedelta(days=6)
        return start.isoformat(), end.isoformat()
    if "last week" in s:
        this_start = today - dt.timedelta(days=today.weekday())
        start = this_start - dt.timedelta(days=7)
        end = start + dt.timedelta(days=6)
        return start.isoformat(), end.isoformat()
    if "this month" in s:
        start = today.replace(day=1)
        if start.month == 12:
            nxt = dt.date(start.year + 1, 1, 1)
        else:
            nxt = dt.date(start.year, start.month + 1, 1)
        end = nxt - dt.timedelta(days=1)
        return start.isoformat(), end.isoformat()
    if "last month" in s:
        first_this = today.replace(day=1)
        end = first_this - dt.timedelta(days=1)
        start = end.replace(day=1)
        return start.isoformat(), end.isoformat()

    m = re.search(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", s
    )
    if m:
        wd = _WEEKDAYS[m.group(1)]
        delta = (today.weekday() - wd) % 7
        if delta == 0:
            delta = 7
        d = today - dt.timedelta(days=delta)
        return d.isoformat(), d.isoformat()

    m = re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", s
    )
    if m:
        wd = _WEEKDAYS[m.group(1)]
        delta = (today.weekday() - wd) % 7
        d = today - dt.timedelta(days=delta)
        return d.isoformat(), d.isoformat()

    return None


# ---------------------------------------------------------------------------
# Lifespan / app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["db"] = lancedb.connect(LANCEDB_PATH)
    state["table"] = open_table(state["db"])
    state["http"] = httpx.AsyncClient()
    state["reranker"] = RRFReranker()
    if state["table"].count_rows() > 0:
        ensure_fts_index(state["table"])
    log.info(
        "rag service ready: lancedb=%s ollama=%s min_score=%s",
        LANCEDB_PATH, OLLAMA_HOST, RAG_MIN_SCORE,
    )
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
    date_from: Optional[str] = Field(None, description="Inclusive lower bound, YYYY-MM-DD")
    date_to: Optional[str] = Field(None, description="Inclusive upper bound, YYYY-MM-DD")


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
    _validate_key(req.key)
    table = state["table"]
    http: httpx.AsyncClient = state["http"]

    new_hash = sha256_hex(req.content)

    # Skip if already indexed with the same hash. This makes /reindex
    # cheap to re-run across the whole vault.
    if existing_hash_for(table, req.key) == new_hash:
        return {"chunks": 0, "skipped": True, "hash": new_hash}

    table.delete(f"key = '{_quote(req.key)}'")

    raw_chunks = chunk_markdown(req.content)
    if not raw_chunks:
        return {"chunks": 0, "skipped": False, "hash": new_hash}

    date, weekday = parse_date_from_key(req.key)

    augmented = [build_chunk_text(date, weekday, section, body) for section, body in raw_chunks]
    vectors = await embed_batch(http, augmented)

    rows = [
        {
            "text": augmented[i],
            "vector": vectors[i],
            "key": req.key,
            "chunk_id": i,
            "content_hash": new_hash,
            "date": date,
            "weekday": weekday,
            "section": raw_chunks[i][0],
        }
        for i in range(len(raw_chunks))
    ]
    table.add(rows)
    if req.rebuild_fts:
        ensure_fts_index(table)
    log.info(
        "indexed %d chunks for %s (date=%s, fts_rebuilt=%s)",
        len(rows), req.key, date or "-", req.rebuild_fts,
    )
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
    # Scan the underlying Lance dataset directly to avoid the unbounded
    # .search().limit() pattern; cost is linear in chunk count.
    arrow_tbl = table.to_lance().to_table(columns=["key", "content_hash"])
    keys = arrow_tbl.column("key").to_pylist()
    hashes = arrow_tbl.column("content_hash").to_pylist()
    by_key: dict[str, str] = {}
    for k, h in zip(keys, hashes):
        if k in by_key:
            continue
        by_key[k] = h or ""
    return {"keys": by_key}


@app.post("/query")
async def query(req: QueryRequest) -> dict[str, list[Hit]]:
    if not req.q.strip():
        raise HTTPException(400, "empty query")

    table = state["table"]
    http: httpx.AsyncClient = state["http"]

    [qvec] = await embed_batch(http, [req.q])

    # Resolve date range: caller-provided wins; otherwise parse from q.
    date_from = (req.date_from or "").strip()
    date_to = (req.date_to or "").strip()
    if not date_from and not date_to:
        rng = parse_temporal_range(req.q)
        if rng:
            date_from, date_to = rng

    where_parts: list[str] = []
    if date_from:
        where_parts.append(f"date >= '{_quote(date_from)}'")
    if date_to:
        where_parts.append(f"date <= '{_quote(date_to)}'")
    where_clause = " and ".join(where_parts) if where_parts else None

    # Fetch a wider net so per-key diversity has something to choose from.
    fetch_n = max(req.k * 4, 30)

    try:
        builder = (
            table.search(query_type="hybrid", vector_column_name="vector")
            .vector(qvec)
            .text(req.q)
            .rerank(reranker=state["reranker"])
            .limit(fetch_n)
        )
        if where_clause:
            builder = builder.where(where_clause)
        results = builder.to_list()
    except Exception as e:
        # Most likely cause: FTS index not yet built. Fall back to
        # dense-only and rebuild for next time.
        log.warning("hybrid search failed (%s), falling back to vector-only", e)
        ensure_fts_index(table)
        builder = table.search(qvec).limit(fetch_n)
        if where_clause:
            builder = builder.where(where_clause)
        results = builder.to_list()

    # Min-score cutoff (only when relevance score available, since
    # _distance has the opposite ordering) + per-key diversity + truncate.
    per_key: dict[str, int] = {}
    hits: list[Hit] = []
    for r in results:
        rel = r.get("_relevance_score")
        if rel is not None:
            score = float(rel)
            if RAG_MIN_SCORE > 0.0 and score < RAG_MIN_SCORE:
                continue
        else:
            score = float(r.get("_distance", 0.0))

        key = r["key"]
        if per_key.get(key, 0) >= MAX_PER_KEY:
            continue
        per_key[key] = per_key.get(key, 0) + 1
        hits.append(
            Hit(
                text=r["text"],
                key=key,
                chunk_id=int(r["chunk_id"]),
                score=score,
            )
        )
        if len(hits) >= req.k:
            break

    return {"hits": hits}
