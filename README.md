# DailyLog

A Telegram bot that turns text and voice messages into a refined daily log inside an Obsidian vault, and answers questions over the whole vault using a local RAG index.

## What it does

- Send any text or voice message; the bot asks Today/Yesterday, refines via Groq into a fixed three-section markdown layout, and writes it to `Daily notes/YYYY/MM/YYYY-MM-DD.md`.
- Send `? <question>` to ask anything about your past notes (vault-wide hybrid search, BM25 + vector, answered by Groq).
- `/reindex` walks the whole vault and embeds every `*.md` into LanceDB. Subsequent runs skip unchanged files via a content hash.

## Stack

- `dailylog` (Go): Telegram bot, refinement, file I/O, RAG client.
- `rag` (Python/FastAPI/LanceDB): chunking, embedding, hybrid retrieval.
- `ollama`: serves `nomic-embed-text` for embeddings.
- Groq: chat completions for refinement and QA.

## Setup

1. Copy `.env.example` to `.env` and fill in:

   | Var | Required | Notes |
   |---|---|---|
   | `TELEGRAM_TOKEN` | yes | BotFather token |
   | `TELEGRAM_CHAT_ID` | yes | Allow-listed chat id (the bot ignores everything else) |
   | `GROQ_API_KEY` | yes | |
   | `GROQ_MODEL` | yes | e.g. `llama-3.3-70b-versatile` |
   | `VAULT_HOST_PATH` | yes | Absolute host path to the Obsidian vault root |
   | `DAILY_NOTES_SUBDIR` | no | Subdir of the vault for daily notes (default `Daily notes`) |
   | `GROQ_TRANSCRIBE_MODEL` | no | Default `whisper-large-v3` |
   | `GROQ_SYSTEM_PROMPT` | no | Override the default refinement prompt |

2. Bring up the stack:

   ```bash
   docker compose up -d
   docker compose logs -f dailylog
   ```

   First boot pulls the embedding model (1-2 min). Wait for `dailylog bot starting...`.

3. In Telegram, send `/reindex` once to backfill the vault.

## Telegram commands

- Plain text or voice note: filed as a daily note after Today/Yesterday selection.
- `?<question>`: hybrid RAG query over the entire vault.
- `/reindex`: walks the vault, embeds new/changed files, skips unchanged ones.
- `/help`: usage summary.

## Where data lives

- Notes: `${VAULT_HOST_PATH}` on the host, mounted at `/vault` in the container.
- LanceDB index: docker volume `dailylog_rag-data` (mounted at `/data` in the rag container).
- Ollama models: docker volume `dailylog_ollama-data`.

## Operational notes

- The sidecar auto-migrates: if the LanceDB schema changes between releases, the table is dropped on startup and the index rebuilds on the next `/reindex`.
- Indexing on save runs in a background goroutine; failures are logged but never block the user.
- Hybrid search uses BM25 (FTS) plus dense (vector) retrieval, fused with reciprocal-rank fusion. The FTS index is rebuilt at startup and after every `/index` insert.
- The bot only responds to `TELEGRAM_CHAT_ID`. Any other chat is silently dropped.
