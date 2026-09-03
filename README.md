# Cleanomatics

A small support assistant for a SaaS company, built with Python and FastAPI.

You send it a question through a `/chat` endpoint and it answers using
information it can actually verify — never guesswork.

## The idea

The assistant handles two kinds of questions:

- **Things that don't change**, like refund rules, billing cycles, and FAQs.
  These live in a local knowledge base of text files that the assistant
  searches through.
- **Things that do change**, like the status of a specific order. These come
  from a tool the assistant calls to look up real data.

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

## Status

Early days — architecture is designed, implementation is next.

## Built with

Python · FastAPI · Groq · pytest
