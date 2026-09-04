# Cleanomatics

A support assistant for **ShipFlow**, built with Python and FastAPI.

You send it a question through a `POST /chat` endpoint and it answers using
information it can actually verify — never guesswork.

## What ShipFlow is

ShipFlow is the product the assistant answers for: a SaaS platform that online
businesses use to manage orders, shipping and tracking, returns and refunds,
subscriptions and billing, and customer accounts.

Its customers ask two very different kinds of question, and telling them apart
is the whole job:

| | Static company knowledge | Dynamic order data |
|---|---|---|
| **What** | Refund rules, delivery times, plan prices, billing policy | Where one specific parcel is, right now |
| **Where it lives** | `knowledge_base/` — eight Markdown documents | The order service, behind `app/tools/orders.py` |
| **How it's reached** | Semantic search over embeddings | A direct lookup by order reference |
| **True for** | Everyone | One order |
| **Changes** | Rarely, when policy changes | Constantly |

*"What is your refund window?"* is the first kind. *"Where is ORD-1001?"* is the
second. *"My order is late, can I get a refund?"* needs both, and the assistant
works out which it needs before answering.

In this project the order service is **mock data** — four seeded orders in a
Python dict, standing in for a real integration. The knowledge base is **static
policy** written as Markdown. Nothing about a specific order is ever in the
knowledge base, and no company policy is ever in the order service.

## Why it works this way

Language models are good at sounding confident and bad at admitting they don't
know something. This project is built the other way around: the model decides
*what to look up* and *how to word the reply*, but every fact in the answer has
to come from a document or a tool result. If nothing relevant turns up, the
assistant says so instead of inventing an answer.

That guarantee is structural, not a matter of asking nicely. The component that
writes the answer cannot reach the knowledge base or the order tool — those
functions are not imported into it. Everything it knows was handed to it.

## Architecture

```
                       POST /chat  {"message": "..."}
                                  │
                    ┌─────────────▼─────────────┐
                    │   app/api/routes.py       │   validates, shapes response
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  app/agent/orchestrator   │   owns failure + confidence
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   PLANNER  (Groq call 1)  │   what does this need?
                    │   temperature 0, JSON     │
                    └─────────────┬─────────────┘
                                  │
                  Plan { needs_kb, needs_order_tool,
                         order_id, kb_query, intent }
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
       ┌─────────▼─────────┐            ┌──────────▼─────────┐
       │  KB RETRIEVAL     │            │   ORDER TOOL       │
       │  embed → cosine   │            │   mock service     │
       │  → threshold      │            │   ORD-#### lookup  │
       │  (static policy)  │            │   (dynamic data)   │
       └─────────┬─────────┘            └──────────┬─────────┘
                 │                                 │
                 └────────────────┬────────────────┘
                                  │
                     Evidence { kb?, order? }   ← verified, or absent
                                  │
                    ┌─────────────▼─────────────┐
                    │  GENERATOR (Groq call 2)  │   no access to either source
                    │  temperature 0.3          │
                    └─────────────┬─────────────┘
                                  │
              {"answer", "sources", "tool_calls", "confidence", "trace"}
```

## Components

| Path | Does |
|---|---|
| `app/api/routes.py` | The `/chat` and `/health` endpoints. No agent logic. |
| `app/main.py` | App wiring, exception handlers, startup warm-up. |
| `app/agent/orchestrator.py` | Joins everything. Owns failure handling and confidence. |
| `app/agent/planner.py` | Groq call 1. Decides what evidence is needed. |
| `app/agent/generator.py` | Groq call 2. Writes the reply from supplied evidence. |
| `app/kb/loader.py` | Reads and cleans the Markdown documents. |
| `app/kb/chunker.py` | Splits them on `##` sections, with context prefixes. |
| `app/kb/embeddings.py` | Local sentence-transformers embeddings. |
| `app/kb/vector_store.py` | In-memory numpy index, cosine similarity. |
| `app/kb/retriever.py` | Ties those together; applies the threshold. |
| `app/tools/orders.py` | Mock order service. Independent of LLM and KB. |
| `app/llm/groq_client.py` | The only module that imports `groq`. |
| `app/llm/prompts.py` | Both system prompts. |
| `app/schemas/` | Pydantic models for every boundary. |
| `knowledge_base/` | Eight ShipFlow policy documents. |

## Request flow

1. **Validate.** `ChatRequest` rejects a missing, empty, blank, or overlong
   message, and any unexpected field, before anything else runs.
2. **Plan.** The question goes to Groq with one job: decide what it needs.
   Returns JSON, validated into a `Plan`. It never answers anything.
3. **Guard.** The plan is not trusted. An order ID absent from the customer's
   message is discarded; one the model missed is recovered by regex; an
   incoherent plan is repaired.
4. **Gather.** If the plan asked for documents, the retriever runs. If it asked
   for an order, the tool runs. Either may fail; both failures are survivable.
5. **Answer.** The evidence goes to Groq with the question, and this call writes
   the reply.
6. **Report.** The answer, its sources, the tool-call count, the confidence, and
   a trace of what happened.

## How the pieces work

**RAG** — retrieval-augmented generation. Rather than relying on what the model
absorbed in training, the relevant policy text is found first and pasted into
the prompt. The model's job becomes reading comprehension over supplied text,
which it is far better at than recall.

**Embeddings** — each chunk of policy is converted into a 384-dimensional vector
by `all-MiniLM-L6-v2`, running locally. No embedding API is called. Texts that
mean similar things land near each other, which is what lets *"when will I see
the money back"* find a document that says *"refund"*.

**Chunking** — documents split on their `##` headings, since a section is
already the unit a person would quote. Sections over 900 characters split again
on paragraph boundaries with 150 characters of overlap. Every chunk is prefixed
with its document title and section heading, so a chunk retrieved alone still
says what it is about.

**Semantic retrieval** — the query is embedded the same way, and cosine
similarity against all 72 chunks is one matrix multiply. Vectors are normalised,
so the dot product *is* the cosine.

**The confidence threshold** — the important part. Nearest-neighbour search
always returns something, however far away. Without a floor, a question the
documents cannot answer would come back with the least-bad paragraph, which
reads exactly like evidence. Matches below `SIMILARITY_THRESHOLD` (0.35) are
dropped, and the result reports `has_evidence: false` instead.

**The order-status tool** — a mock external service over four seeded orders. It
returns a structured result for every outcome: found, not found, no reference
given, or service failure. An unknown reference is reported as unknown, never
approximated to a nearby order. `FAIL-001` deliberately raises, so failure
handling can be exercised.

**The planner (Groq call 1)** — reads the question and writes a shopping list.
It is told what the two sources contain and asked which this question needs.
Temperature 0, JSON mode. It has no field it could put an answer in.

**The answer generator (Groq call 2)** — receives the question and the evidence,
and writes the reply. It cannot search or look anything up. If the evidence
doesn't support an answer, it says so.

## Project structure

```
cleanomatics/
├── app/
│   ├── main.py                  FastAPI app, exception handlers
│   ├── config.py                all settings, from the environment
│   ├── api/routes.py            /chat and /health
│   ├── agent/
│   │   ├── orchestrator.py      plan → gather → answer
│   │   ├── planner.py           Groq call 1
│   │   └── generator.py         Groq call 2
│   ├── kb/
│   │   ├── loader.py            read and clean documents
│   │   ├── chunker.py           split into retrievable pieces
│   │   ├── embeddings.py        local vectors
│   │   ├── vector_store.py      in-memory index
│   │   └── retriever.py         semantic search + threshold
│   ├── llm/
│   │   ├── groq_client.py       the only file that imports groq
│   │   ├── prompts.py           both system prompts
│   │   └── errors.py            structured LLM errors
│   ├── tools/
│   │   ├── orders.py            mock order service
│   │   └── errors.py            structured tool errors
│   └── schemas/                 chat, agent, kb, orders, trace
│   └── static/index.html        the chat interface, one file
├── knowledge_base/              8 ShipFlow policy documents
├── scripts/
│   ├── search_kb.py             search the KB from the command line
│   ├── demo_llm.py              drive the pipeline by hand
│   └── list_models.py           what models this Groq key can use
├── tests/                       10 test modules
├── requirements.txt
├── .env.example
└── pyproject.toml
```

## Setup

```bash
git clone https://github.com/asitgiri1234/cleanomatics.git
cd cleanomatics

python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

The first run downloads the embedding model (about 90 MB) and caches it.

## Environment variables

Copy the example file and paste your Groq key into it:

```bash
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

| Variable | Default | What it does |
|---|---|---|
| `GROQ_API_KEY` | *(none)* | Your Groq key. Required for `/chat`. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model for both calls. |
| `GROQ_TIMEOUT_SECONDS` | `30` | Per-request timeout. |
| `GROQ_MAX_RETRIES` | `2` | SDK-level retries. |
| `LLM_MAX_TOKENS` | `1024` | Cap on reply length. |
| `KB_DIR` | `knowledge_base` | Where the documents live. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model. |
| `TOP_K` | `4` | Chunks retrieved per query. |
| `SIMILARITY_THRESHOLD` | `0.35` | Floor for usable evidence. |
| `CHUNK_MAX_CHARS` | `900` | Section split size. |
| `CHUNK_OVERLAP_CHARS` | `150` | Overlap between split pieces. |
| `ALLOWED_ORIGINS` | `*` | CORS origins. Only needed if the UI is served separately. |

Get a key at https://console.groq.com/keys. `.env` is gitignored and never
committed. An environment variable of the same name overrides the file.

Groq retires models periodically. If you see a configuration error naming the
model, run `python scripts/list_models.py` and update `GROQ_MODEL`.

Retrieval and the order tool work without a key — only the two LLM calls need
one.

## Running the server

```bash
uvicorn app.main:app --reload
```

| URL | What |
|---|---|
| http://127.0.0.1:8000/ | **Chat interface** |
| http://127.0.0.1:8000/docs | Swagger docs |
| http://127.0.0.1:8000/health | Readiness check |

## The chat interface

One static HTML file at `app/static/index.html`, served by the same FastAPI app
that answers `/chat`. No framework, no build step, no second server — and
because the page and the API share an origin, no CORS to configure.

It shows the conversation, a loading indicator while Groq is thinking, and a
collapsed panel under each answer with the sources, confidence, tool-call count,
whether retrieval ran, the best match score against the threshold, LLM calls,
and any degraded steps. All of that comes straight from the `/chat` response —
the page computes nothing and contains no agent logic.

API errors are rendered in the conversation rather than thrown away, so a 503
from a missing key or a 502 from Groq reads as a message instead of a silent
failure.

`ALLOWED_ORIGINS` exists for the case where you serve the page from somewhere
else; the bundled UI does not need it.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What is the status of ORD-1001?\"}"
```

## Example response

```json
{
  "answer": "Your order ORD-1001 shipped on 3 September with FastPost, tracking number FP884213099US, and is estimated to arrive on 8 September.",
  "sources": ["order:ORD-1001"],
  "tool_calls": 1,
  "confidence": 1.0,
  "trace": {
    "intent": "order status",
    "kb_used": false,
    "kb_calls": 0,
    "kb_confidence": 0.0,
    "kb_threshold": 0.0,
    "order_tool_used": true,
    "tool_calls": 1,
    "order_outcome": "found",
    "llm_calls": 2,
    "sources": ["order:ORD-1001"],
    "confidence": 1.0,
    "grounded": true,
    "degraded": []
  }
}
```

The first four fields are the response contract. `trace` is observability — it
shows which components ran and how well supported the answer was. It carries no
prompts, no chunk text, and no secrets.

Errors use one shape, at 422 (invalid request), 502 (model failed), 503 (not
configured), 504 (timed out), or 500 (unexpected):

```json
{"error": {"code": "invalid_request", "message": "The request was not valid: 'message' Field required. Send {\"message\": \"your question\"}."}}
```

## Running tests

```bash
python -m pytest                                  # the offline suite
python -m pytest -v                               # with test names
python -m pytest tests/test_scenarios.py -v       # the six required scenarios
```

Integration tests call the real Groq API and are deselected by default:

```bash
python -m pytest tests/test_groq_integration.py -m integration -v
```

They skip cleanly if `GROQ_API_KEY` is unset. Everything else runs offline
against a fake client, so the suite is deterministic and needs no key.

Command-line tools, no server required:

```bash
python scripts/search_kb.py --demo                # retrieval only, no Groq
python scripts/demo_llm.py --demo                 # the full pipeline
```

## The six required scenarios

Each runs deterministically in `tests/test_scenarios.py` and again against live
Groq in `tests/test_groq_integration.py`.

Measured against a live server and the real Groq API:

| # | Scenario | KB used | Tool calls | Confidence | Sources |
|---|---|---|---|---|---|
| 1 | `"What is ShipFlow's refund policy?"` | yes | 0 | 0.74 | `refund-policy.md` |
| 2 | `"What is the status of ORD-1001?"` | no | 1 | 1.00 | `order:ORD-1001` |
| 3 | `"What is your refund policy and what is the status of ORD-1001?"` | yes | 1 | 0.87 | both |
| 4 | `"What shift patterns do your warehouse staff work?"` | no | 0 | 0.00 | none |
| 5 | `"What is the status of FAIL-001?"` | no | 1 | 0.00 | none |
| 6 | `"Tell me about my order."` | no | 0 | 0.00 | none |

What each demonstrates:

1. **KB only.** The planner asked for documents and no order. Answered from
   `refund-policy.md` at 0.74 similarity, with the source named.
2. **Tool only.** No retrieval ran at all — the planner recognised this needs
   only the order record. *"Your order ORD-1001 has been shipped. It's on its
   way via FastPost (tracking FP884213099US) and is expected 2026-09-08."*
3. **Both.** One retrieval and one tool call; the reply states the refund rules
   and the order's status together. Confidence 0.87 is the mean of 0.74 and 1.0.
4. **Unsupported.** The planner judged that nothing in the documents covers
   warehouse rosters, so nothing was retrieved and confidence is 0.00. *"I'm
   sorry, but I couldn't confirm the shift patterns for our warehouse staff.
   Please reach out to ShipFlow support."* No fact was asserted.
5. **Tool failure.** `FAIL-001` raised inside the service; `lookup_order` turned
   it into a `service_error` outcome. The API still returned 200 with a
   controlled reply and `degraded: ["order_service_error"]`. No order details
   were invented — the generator was explicitly told not to describe one.
6. **Ambiguous.** No reference in the message, so the tool was never called.
   *"I'd be glad to check on that. Could you give me your order reference?"*

Scenario 6 is worth a note. The planner is not deterministic about whether this
question needs the documents; on some runs it sets `needs_kb`. Either way the
customer is asked for the reference, because the orchestrator supplies the
lookup's "no ID supplied" outcome whenever an order question arrives without
one. Without that, a run that searched the documents would explain what was
missing but never actually ask.

## Known limitations and assumptions

**Retrieval is sensitive to phrasing.** Heavily colloquial questions retrieve
poorly. *"i changed my mind, can i send this back"* scores about 0.31 against
every document, and its closest match is the wrong one. The planner's rewrite
(`return an item`, 0.63 on `returns.md`) is what fixes this, which is why
retrieval runs on `plan.kb_query` and never on the raw message. Recorded in
`tests/test_retriever.py`.

**The threshold is a coarse filter, not a relevance judge.** At 0.35 it reliably
rejects off-topic questions, but topically adjacent ones can still clear it —
`who is your CEO` scores 0.348 against the ownership-transfer section. The
generator's grounding instruction is the second line of defence, and in practice
it is the one that catches these. The number was measured, not guessed: correct
retrievals cluster at 0.40–0.77, off-topic ones at 0.08–0.35.

**Confidence measures evidence support, not correctness.** It is the mean of the
scores for the sources the plan asked for. A found order counts as 1.0, since
the order service is the system of record. A plan that asked for nothing — a
greeting — scores 0.0, which is honest rather than a failure.

**No conversation memory.** Each request is independent. A follow-up like
*"what about that one?"* has no antecedent to resolve.

**The order service is mock data.** Four orders in a dict. A real integration
would add network failures, auth, and pagination, which is why `lookup_order()`
already returns a structured `SERVICE_ERROR` rather than assuming success.

**Order data is per-order, policy is universal.** The assistant never infers
policy from an order record, or order state from policy. If a customer asks
something the documents don't cover about an order the tool doesn't have, it
declines both halves separately.

**Planner quality is model-dependent.** The guards catch invented and missed
order IDs, but a planner that misjudges *whether* a question needs the knowledge
base will produce a thinner answer. Nothing detects that automatically.

**Single-process, in-memory.** The index is rebuilt at startup, about two
seconds. Fine for one process; a multi-worker deployment rebuilds it per worker.

## Built with

Python · FastAPI · Groq · sentence-transformers · numpy · pytest
