
# SHL Conversational Assessment Recommender

## Overview

The SHL Conversational Assessment Recommender is a stateless, production-grade FastAPI service designed to turn vague hiring requirements into a grounded shortlist of SHL **Individual Test Solutions** through multi-turn dialogue[cite: 1]. Engineered with advanced Retrieval-Augmented Generation (RAG) and robust stateless orchestration, it strictly maps conversational hiring needs to available catalog data while enforcing hard conversational guardrails against hallucination and out-of-scope queries[cite: 1].

## Deployment

Application endpoint link - [https://shl-assessment-recommender-production-a2cc.up.railway.app](<Backend Endpoint>)

---

## 🏗️ Architecture & Tech Stack

This system leverages Clean Architecture principles with a strictly modular pipeline. It is entirely stateless; rather than pinning user sessions to a persistent database, the client maintains the conversation state and re-submits it with each API call.

| Layer                      | Technology                                                       | Function                                                                 |
| :------------------------- | :--------------------------------------------------------------- | :----------------------------------------------------------------------- |
| **API Framework**    | FastAPI + Uvicorn                                                | Handles routing and stateless request validation.                        |
| **LLM & Embeddings** | Gemini API (`gemini-2.5-flash-lite`, `gemini-embedding-001`) | Powers intent classification, query extraction, and response generation. |
| **Vector Database**  | Qdrant + BM25                                                    | Facilitates hybrid semantic (dense) and lexical (sparse) retrieval.      |
| **Deployment**       | Railway + Docker                                                 | Ensures robust, containerized cloud deployment.                          |

### Core Pipeline Layers

* **Intent Classification:** Dynamically determines if the user is clarifying, refining, comparing, or recommending to inject specific prompt instructions.
* **Constraint Extraction:** Distills job role, seniority, skills, and languages from the conversation history.
* **In-Memory Catalog Loading:** Parses the catalog and instantiates the embeddings/Qdrant index into memory on startup to eliminate cold-start retrieval latency.
* **State Reconstruction:** Re-derives the "current shortlist" on every call by parsing the agent's own prior structured output from the conversation history[cite: 1].
* **Post-Hoc Validator:** Programmatically checks every generated URL against the catalog index to guarantee zero hallucinated links[cite: 1].

---

## 🚀 Advanced Retrieval Strategy

To combat the limitations of pure vector search—which often over-indexes on narrow technical terms at the expense of foundational cognitive/behavioral tests—the system employs specialized retrieval interventions:

* **Hybrid Core with RRF:** Fuses Gemini dense embeddings with BM25 exact-keyword hits using Reciprocal Rank Fusion.
* **Diversity Deduplication:** Fetches 4x the required candidates, normalizes names, and strips variants to prevent the LLM from being flooded with identical tests.
* **Compare Intent Bypass:** Intelligently overrides the diversity deduplicator when a user asks to compare variants (e.g., "What is the difference between MS Excel and MS Excel 365?").
* **Pinned Heuristics:** Unconditionally pins baseline behavioral and cognitive tests (like OPQ32r and SHL Verify Interactive G+) into the generation pool so they aren't lost to technical keyword dominance.
* **Companion Co-Retrieval:** Automatically injects companion contextual reports (Leadership, Sales) into the context whenever a foundational assessment like OPQ32r is retrieved.
* **Regex Priority Overrides:** Forces explicitly named acronyms (e.g., "GSA") to the top of the context window with the highest priority, bypassing semantic sorting.

---

## 💬 Conversational Behaviors

The system adheres strictly to defined behavioral boundaries[cite: 1]:

* **Clarify:** The agent asks one targeted question at a time to determine role, seniority, purpose, or delivery constraints before emitting a shortlist[cite: 1].
* **Recommend:** Returns 1–10 items strictly sourced from the retrieval index[cite: 1]. The agent employs a transparency pattern for default inclusions (e.g., adding OPQ32r for personality testing) and explicitly notes trade-offs when constraints conflict with test availability[cite: 1].
* **Refine:** Reconstructs the current shortlist and applies only the requested user delta[cite: 1]. Unaffected items remain byte-identical across turns[cite: 1]. The agent may push back once on an illogical request but must comply if the user repeats the instruction[cite: 1].
* **Compare:** Answers are strictly grounded in retrieved catalog descriptions and differences (e.g., instrument vs. report, legacy vs. new variants) rather than external LLM knowledge[cite: 1].
* **Refuse:** Deflects off-topic queries, legal/compliance advice, and prompt injections[cite: 1]. Refusals specifically state what is out of scope while redirecting the user appropriately, keeping the session open for valid requests[cite: 1].

---

## 📡 API Contract

The application exposes two primary routes[cite: 1]:

### `GET /health`

* Returns `{"status": "ok"}`[cite: 1].
* Tolerates up to 2 minutes of cold-start latency on the first call[cite: 1].

### `POST /chat`

Accepts a JSON payload containing the `messages` array[cite: 1]. The system operates under a strict 30-second per-call timeout budget and a maximum conversation cap of 8 turns[cite: 1].

**Response Schema Requirements:**

* **`reply`:** (String) The conversational response to the user[cite: 1].
* **`recommendations`:** (Array or Null) Contains 1–10 objects with `name`, `url`, and `test_type` when committing to a shortlist, otherwise `null` during clarification or refusal[cite: 1].
* **`end_of_conversation`:** (Boolean) Set to `true` only when the task is complete and a shortlist is delivered without open questions[cite: 1].

---

## 🛡️ Fallback Logic & Graceful Degradation

Because the system is stateless and evaluated by a strict automated harness, it employs rigorous fallback chains to ensure a schema-valid response is always returned[cite: 1]:

* **Data Layer:** If the live SHL JSON catalog endpoint fails, the system loads the most recent successfully fetched local snapshot from disk[cite: 1]. Ambiguous records regarding Individual vs. Job Solution classification are excluded to prioritize safety[cite: 1].
* **Retrieval Layer:** If the dense embedding index errors out, the system falls back to BM25 lexical-only retrieval[cite: 1]. If zero results are found, the agent returns a clarifying reply rather than fabricating a shortlist[cite: 1].
* **LLM Layer:** If the LLM generates malformed JSON, the system validates and retries[cite: 1]. If hallucinated URLs are detected, they are stripped; if this leaves zero valid items on a commit turn, the system falls back to top-K retrieved items templated into a static reply[cite: 1].
* **Statelessness Layer:** If a prior shortlist cannot be confidently reconstructed from history during a refine turn, the system runs a fresh retrieval against all accumulated constraints rather than guessing silently[cite: 1].

---

## 🧪 Evaluation Methodology

Evaluation is cleanly decoupled into two strictly defined paradigms:

* **Retrieval Purity (Recall@10):** A simulated user replays a ground-truth dataset (`C1–C10` conversation traces) to measure if the target catalog items appeared inside the LLM's final commitment[cite: 1].
* **Hard Evals:** Validates JSON schema adherence, enforces that every URL exists exactly in the scraped catalog index, and ensures the conversation never exceeds the 8-turn cap[cite: 1].
* **Behavior Probes:** Pytest assertions hit live endpoints dynamically to verify stability during refine turns, legal/compliance refusal routing, and prompt-injection resistance[cite: 1].

---

## 📁 Directory Structure

```text
SHL/
├── data/
│   ├── catalog_cache.json
│   ├── faiss_index.bin
│   └── bm25_corpus.pkl
├── scripts/
│   └── ingest_catalog.py
├── src/
│   ├── main.py
│   ├── config/
│   ├── domain/
│   ├── catalog/
│   ├── retrieval/
│   ├── orchestration/
│   ├── llm/
│   ├── prompts/
│   └── api/
├── tests/
│   ├── unit/
│   └── eval/
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```
