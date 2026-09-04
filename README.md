# Cleanomatics

A support assistant for **ShipFlow**, built with Python and FastAPI.

You send it a question through a `/chat` endpoint and it answers using
information it can actually verify — never guesswork.

## What ShipFlow is

ShipFlow is the product the assistant answers for: an order and delivery
service with customers who ask about policies ("what's your refund window?")
and about their own orders ("where is order 1042?"). Two very different kinds
of question, and the assistant has to tell them apart.

- **Things that don't change** — refund rules, billing cycles, shipping
  policies, FAQs. These live in a local knowledge base of text files under
  `knowledge_base/` that the assistant searches through.
- **Things that do change** — the status of a specific order. These come from
  a ShipFlow order-lookup tool the assistant calls for real data.

Some questions need one, some need the other, and some need both. For example,
*"my order is late, can I get a refund?"* needs the order lookup **and** the
refund policy. The assistant works out which it needs before answering.

## Why it matters

Language models are good at sounding confident and bad at admitting they don't
know something. This project is built the other way around: the model decides
*what to look up* and *how to word the reply*, but every fact in the answer has
to come from a document or a tool result. If nothing relevant turns up, the
assistant says so instead of inventing an answer.

Every reply also lists where its information came from, so you can check it.

## How it works

A question makes **two Groq calls**, with the retrieval and tool work sitting
between them:

1. **Planner call.** The question goes to Groq with one job: decide what this
   question needs — knowledge-base lookup, an order lookup, both, or neither —
   and pull out any order ID it mentions. It returns a small structured plan.
   It does not answer anything.
2. **Gathering.** The plan is executed in ordinary Python. If it asked for
   documents, the retriever embeds the question and finds the closest chunks
   from the knowledge base. If it asked for an order, the order tool is called.
3. **Answer call.** The gathered context goes back to Groq with the original
   question, and this call writes the reply — grounded in that context only,
   with its sources listed.

Splitting it this way keeps each call simple: one decides, one writes. Neither
is asked to do both at once.

## Layout

```
app/
  main.py          FastAPI app
  config.py        settings from the environment
  api/             the /chat endpoint
  schemas/         Pydantic request, response, and internal models
  agent/           the planner call and the orchestration around it
  kb/              loading, chunking, embedding, and searching the documents
  llm/             Groq client and prompt templates
  tools/           the ShipFlow order lookup
knowledge_base/    ShipFlow policy documents
scripts/           standalone command-line tools
tests/
```

## Searching the knowledge base

Retrieval works on its own, with no Groq key and no server running:

```bash
python scripts/search_kb.py "how long do refunds take"
python scripts/search_kb.py --demo
```

Documents are split on their Markdown sections, each chunk is embedded locally
with `all-MiniLM-L6-v2`, and search is cosine similarity against those vectors
held in memory. Anything scoring below `SIMILARITY_THRESHOLD` is dropped, so a
question the documents do not cover comes back with no evidence rather than
with the closest available paragraph.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in GROQ_API_KEY
uvicorn app.main:app --reload
```

## Status

Knowledge base and retrieval work. The Groq calls, the order lookup, and the
`/chat` endpoint are not built yet.

## Built with

Python · FastAPI · Groq · sentence-transformers · pytest
