# SHL Conversational Assessment Recommender

## Overview
A stateless conversational agent built with FastAPI that translates vague hiring needs into a grounded shortlist of SHL Individual Test Solutions through a multi-turn dialogue. The architecture is built entirely to ensure high-fidelity, highly relevant product recommendations by combating the typical limitations of pure vector search.

## 🚀 Key Innovative Approaches (Why This Assignment Stands Out)

### 1. Advanced Hybrid Retrieval (RRF) with Graceful Degradation
- **Dense + Sparse Fusion:** Leverages Qdrant for semantic capabilities (via Gemini embeddings) and client-side BM25 for precise keyword hits, merging both signals using Reciprocal Rank Fusion (RRF) to get the best of both worlds.
- **Resilient Fallbacks:** Ensures that if the dense embedder or Qdrant fails, the system smoothly falls back to a sparse-only retrieval rather than crashing, preventing query dropouts.
- **FastAPI Lifespan Caching:** The product catalog is ingested and indexed exactly *once* during application startup. After startup, all requests are served entirely from in-memory state and the vector index, ensuring zero cold-start latency on individual queries.

### 2. Intelligent Candidate Diversity & Intent Detection
- **Diversity Re-Ranking (Deduplication):** A common RAG issue is vector search returning 5 nearly identical versions of the same product, squeezing out diversity. The system automatically normalizes candidate names and deduplicates near-identical hits (e.g., stripping version markers) before passing candidates to the LLM.
- **Compare Intent Bypass:** The system features regex-driven intent detection to recognize comparison queries (e.g., *"difference between X and Y"*). When a compare intent is detected, the diversity filter is intelligently overridden so the LLM receives the similar variants (e.g. *"MS Excel (New)"* vs *"Microsoft Excel 365 (New)"*) and can conduct a nuanced, side-by-side evaluation.

### 3. Smart Heuristic Injection (Pinned Context)
- **Baseline Inclusions:** Pure semantic models often over-index heavily on narrow technical keywords (like "AWS") and neglect foundational behavioral or cognitive baseline tests. The system mitigates this by intentionally pinning robust baselines—like **OPQ32r (Personality)** and **SHL Verify Interactive G+ (Cognitive)**—into the context window as candidates. It dynamically bumps the lowest-confidence natural hits to make room, ensuring reliable baseline recommendations don't get lost in keyword wars.
- **Companion Report Co-Retrieval:** The system recognizes product ecosystems. If OPQ32r successfully makes it into the candidate pool, the retriever automatically injects its companion contextual reports (Leadership, Sales, Universal Competency) into the context for the LLM. This prevents "bare" test recommendations and allows the LLM to proactively suggest powerful add-ons.

### 4. Precision Regex Entity Matching
- **Overriding the Vectors:** When a user explicitly mentions a branded acronym or exact catalog name (e.g., `"GSA"`, `"Contact Center Call Simulation"`), the Regex Entity Matcher catches it. These explicitly mentioned items are forced to the top with the absolute highest priority, bypassing standard relevance scoring completely. This ensures direct user directives are fulfilled immediately.

### 5. Advanced Stateless Pipeline Orchestration
- **Modular LLM Pipeline:** The conversational brain operates on distinct functional phases: an `intent_classifier` to understand user strategy, a `constraint_extractor` to distill requirements, and a `shortlist_parser` to cleanly format the final test offerings.
- **Stateless Tracking:** The API does not rely on heavy persistent session databases for conversational logic. Conversations and turn counting (`turn_counter.py`) can scale smoothly, relying dynamically on the context structure provided by the client per request.

### 6. Production-Ready Deployment configuration
- Engineered to be seamlessly deployed to platforms like Railway, leveraging a custom decoupled `Dockerfile` that ensures the build dependencies operate reliably, mapping environment variables efficiently (`$PORT`) under rigid CI/CD environments.
