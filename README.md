# Multi-Agent Document RAG

**Ask the report a question — then make a second agent check whether the answer actually holds up.**

Built with LangGraph, this project coordinates three specialized AI agents for document question answering: one checks relevance, one researches and drafts the answer, and one verifies every claim against the source material.

The system is built around a deliberately simple idea: retrieval and generation are not enough on their own. Before an answer reaches the UI, the workflow first checks whether the uploaded material is relevant, generates an answer only from retrieved context, and runs a separate verification pass against the same source material.

## The route a question takes

```text
uploaded files
      │
      ▼
Docling → Markdown chunks → cache
      │
      ▼
BM25 + Chroma vector search
      │
      ▼
relevance checker ── NO_MATCH ──▶ reject the question
      │
      │ CAN_ANSWER / PARTIAL
      ▼
research agent → draft answer
      │
      ▼
verification agent
      │
      ├── supported and relevant ──▶ return answer + report
      └── unsupported ─────────────▶ research again
```

The three model calls have separate responsibilities:

- `RelevanceChecker` decides whether the retrieved passages can answer the question.
- `ResearchAgent` writes an answer using only those passages.
- `VerificationAgent` looks for unsupported claims, contradictions, and irrelevant content.

LangGraph coordinates the loop. LangChain message templates keep instructions separate from user questions and document text.

## Retrieval is intentionally hybrid

Semantic search is useful when the wording changes; keyword search is useful when the wording must not change — model names, dates, acronyms, efficiency values, and table labels are common examples. This project combines both:

- BM25 weight: `0.4`
- Chroma vector-search weight: `0.6`
- OpenAI embeddings: `text-embedding-3-small`
- Generation and verification: `gpt-4.1-mini`

The defaults live in [`config/settings.py`](config/settings.py).

## Run it locally

Requirements:

- Python 3.12+
- an OpenAI API key
- [`uv`](https://docs.astral.sh/uv/) or `pip`

With `uv`:

```bash
git clone https://github.com/patrykn11/multi-agent-document-rag.git
cd multi-agent-document-rag
uv sync
```

Create a `.env` file:

```dotenv
OPENAI_API_KEY=your_key_here
```

Start the app:

```bash
uv run python app.py
```

## Supported documents

The uploader accepts:

- PDF
- DOCX
- Markdown
- plain text

Individual files are limited to 50 MB and the combined upload to 200 MB. Parsed chunks are cached for seven days in `document_cache/`; vector data is stored in `chroma_db/`. Both directories are ignored by Git.

## Repository map

```text
agents/               relevance, research, verification, and LangGraph workflow
document_processor/   Docling conversion, Markdown splitting, validation, and cache
retriever/            BM25 and Chroma ensemble construction
config/               file limits, retrieval weights, and environment settings
examples/             reports that can be loaded directly from the UI
app.py                Gradio interface and session-level retriever reuse
```

## What the verification report means

Each answer is accompanied by a compact report:

```text
Supported: YES/NO
Unsupported Claims: [...]
Contradictions: [...]
Relevant: YES/NO
Additional Details: ...
```
