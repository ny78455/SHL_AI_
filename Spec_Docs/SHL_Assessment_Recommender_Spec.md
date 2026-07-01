# SHL Conversational Assessment Recommender — Detailed Technical Specification

Derived from: `SHL_AI_Intern_Assignment.pdf`, the sample catalog export, and the 10 labeled conversation traces (`C1–C10`).

---

## 1. Purpose & Framing

Build a stateless, conversational agent that turns a vague hiring need ("I'm hiring a Java developer") into a grounded shortlist of 1–10 SHL **Individual Test Solutions**, through multi-turn dialogue. The agent must clarify before recommending, revise a shortlist in place when constraints change, answer comparison questions strictly from catalog data, and refuse anything outside SHL assessment selection.

This document specifies the data model, system architecture, API contract, conversational behavior rules, prompt/context strategy, and evaluation plan needed to implement and defend the assignment.

---

## 2. Scope

**In scope**
- SHL product catalog at `https://www.shl.com/solutions/products/product-catalog/`, filtered to **Individual Test Solutions only**.
- A `POST /chat` endpoint that is fully stateless (client resends full history each call).
- Four behaviors: Clarify, Recommend, Refine, Compare.
- Scope refusal: off-topic, legal/compliance advice, general hiring advice, prompt injection.

**Out of scope**
- Pre-packaged Job Solutions (bundled multi-assessment products framed as "solutions" rather than individual tests).
- Any recommendation not traceable to a scraped catalog URL.
- Persisting conversation state server-side.
- General HR/legal advisory content (e.g., "are we legally required to test under HIPAA?" → refuse, per `C7` Turn 3).

---

## 3. Catalog Data Model

### 3.0 Primary data source

A full JSON dump of the catalog export is available directly at:

```
https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json
```

This is the same record shape as the sample export (§3.1) but the complete set — confirmed by fetching it directly, it returns hundreds of records (`.NET Framework 4.5`, `Core Java (Advanced Level) (New)`, `Docker (New)`, `Dependability and Safety Instrument (DSI)`, etc., all present with the same fields). **Treat this JSON endpoint as the primary ingestion source instead of scraping the catalog UI page-by-page** — it is faster, less brittle (no HTML parsing, no pagination/JS-rendering concerns), and already structurally normalized (`job_levels`, `languages` pre-split into arrays). Building an HTML scraper as in §3.3/§10 should be a **fallback path only**, used if this JSON endpoint becomes unavailable, changes shape, or is found to be incomplete/stale relative to the live site (verify record counts and a sample of URLs against the live catalog page during development to confirm parity before trusting it as sole source — see §12.1 for the fallback chain if it isn't).

Important caveat found on inspection: **this JSON dump does not carry an explicit flag distinguishing "Individual Test Solutions" from "Pre-packaged Job Solutions."** It contains records that read as clearly individual (`.NET Framework 4.5`, `Core Java (Advanced Level) (New)`) alongside records that read as bundled job solutions by name and description (e.g., `Entry Level Cashier Solution`, `Entry Level Sales Solution`, `Entry Level Technical Support Solution`, `Customer Service Phone Solution` — each described as "is for entry-level positions in which employees..." and each covering multiple `keys` at once, the classic signature of a packaged solution). This directly affects the assignment's explicit scoping instruction to use Individual Test Solutions only, and needs deliberate handling — see §3.3 and §12.1.

### 3.1 Raw scraped record (per assessment)

Based on the sample export, each catalog entry has this shape:

```json
{
  "entity_id": "3827",
  "name": ".NET Framework 4.5",
  "link": "https://www.shl.com/products/product-catalog/view/net-framework-4-5/",
  "scraped_at": "2026-05-08T10:39:46.810448+00:00",
  "job_levels": ["Professional Individual Contributor", "Mid-Professional"],
  "job_levels_raw": "Professional Individual Contributor, Mid-Professional,",
  "languages": ["English (USA)"],
  "languages_raw": "English (USA),",
  "duration": "30 minutes",
  "duration_raw": "Approximate Completion Time in minutes = 30",
  "status": "ok",
  "remote": "yes",
  "adaptive": "yes",
  "description": "The .NET Framework 4.5 test measures knowledge of .NET environment...",
  "keys": ["Knowledge & Skills"]
}
```

| Field | Type | Notes |
|---|---|---|
| `entity_id` | string | Stable catalog ID; primary key for retrieval and dedup. |
| `name` | string | Display name. Often carries a "(New)" suffix distinguishing a newer variant from a legacy one of near-identical name — **do not collapse these into one entity** (see §7.4). |
| `link` | string (URL) | The **only** URL the agent is allowed to return. Must be scraped, never constructed/guessed. |
| `job_levels` | string[] | Parsed from `job_levels_raw`. Used as a hard/soft filter on seniority. |
| `languages` | string[] | Parsed from `languages_raw`. Can be empty (e.g., legacy report-only products with no timed language delivery). Used for language-fit filtering (`C7`). |
| `duration` | string | Human string ("30 minutes", "Untimed", "Variable", or `""`). Not always numeric — parse defensively. |
| `status` | string | Scrape health flag; filter to `"ok"` records only. |
| `remote` | "yes"/"no" | Remote-proctoring/administration flag. |
| `adaptive` | "yes"/"no" | Whether the test is CAT (computer-adaptive). |
| `description` | string | Free text, primary source for semantic retrieval and for Compare answers. |
| `keys` | string[] | The **category taxonomy**, e.g. `Ability & Aptitude`, `Assessment Exercises`, `Biodata & Situational Judgment`, `Competencies`, `Development & 360`, `Personality & Behavior`, `Knowledge & Skills`, `Simulations`. This is distinct from `test_type`. |

### 3.2 The `test_type` field (from conversation traces, not the raw scrape)

The API response schema requires `test_type` per recommendation. Cross-referencing the traces against `keys`, `test_type` is SHL's **single/multi-letter code** shown in their public catalog UI, mapped roughly:

| Code | Category (≈ `keys`) |
|---|---|
| A | Ability & Aptitude |
| B | Biodata & Situational Judgment |
| C | Competencies |
| D | Development & 360 |
| K | Knowledge & Skills |
| P | Personality & Behavior |
| S | Simulations |
| E | Assessment Exercises (seen in some SHL catalogs; verify against live scrape) |

Products can carry multiple codes (e.g., `K,S` for Microsoft Word 365 (New), `B,S` for Customer Service Phone Simulation, `C,K` for Global Skills Assessment, `P,C` for Entry Level Customer Serv). **The scraper must capture this code directly from the live site** (it's typically a badge/label near duration on the product page) rather than inferring it purely from `keys`, since the mapping is not strictly 1:1 (e.g., `Global Skills Development Report` has `keys` spanning six categories but a single `D` type badge).

### 3.3 Filtering to "Individual Test Solutions"

The SHL catalog UI splits into two tabs: **Pre-packaged Job Solutions** and **Individual Test Solutions**. The JSON dump in §3.0 does not carry this tab distinction as a field, so it must be reconstructed. Recommended approach, in priority order:

1. **Cross-reference against the live catalog UI's Individual Test Solutions tab** (the different URL path/filter parameter SHL exposes on the product-catalog page) to build a set of canonical `entity_id`s or `link`s that belong to that tab, and treat that set as the authoritative membership list — this is more reliable than any heuristic on the JSON fields alone, since it's the actual ground truth the assignment is scoped to.
2. If a full crawl of the tab-filtered UI isn't feasible in the time available, fall back to a **rule-based classifier** over the JSON fields, in this order of confidence:
   - A record whose `keys` is a single category (e.g., only `Knowledge & Skills`, or only `Personality & Behavior`) is very likely an Individual Test Solution.
   - A record whose `keys` spans 2+ categories *and* whose `name`/`description` uses bundling language ("Solution" in the name; description phrased as "is for entry-level positions in which employees..." describing a job archetype rather than a single construct) is very likely a Pre-packaged Job Solution — exclude it.
   - Ambiguous middle cases (e.g., `Global Skills Development Report`, `keys` spanning six categories but functioning as a single report product, seen used as an individual recommendation in `C5`) should be resolved by checking the live product page's own tab placement, not guessed from field patterns alone.
3. Drop any entity with `status != "ok"`.
4. Persist `scraped_at`/fetch timestamp for staleness tracking; re-fetch periodically since SHL updates product names/links (e.g., "(New)" launches, and the JSON endpoint itself could change or move).
5. Log and manually review the classifier's boundary cases before deployment — misclassifying a Job Solution as an Individual Test (or vice versa) directly fails the "items from catalog only in recommendations" hard eval if the evaluator checks tab membership, and silently degrades Recall@10 either way.

### 3.4 Storage & indexing

- **System of record**: a flat JSON/Parquet file or SQLite table of all catalog records, keyed by `entity_id`, rebuilt on each scrape run — this is the retrieval corpus, not a live web call at inference time (inference must stay inside the 30-second per-call budget).
- **Retrieval index**: build both
  - a **dense embedding index** (e.g., FAISS/Chroma) over a concatenation of `name + description + keys + job_levels`, for semantic matching of vague/paraphrased queries ("stakeholder-facing Java dev" → Java + soft-skill/communication signals), and
  - a **lexical/BM25 or exact-match index** over `name`, `entity_id`, and category codes, for precise lookups when the user or Compare turn names a specific product ("OPQ32r", "GSA").
- Hybrid retrieval (dense + lexical, reciprocal-rank fusion) is recommended: pure embedding search under-performs on exact product-name queries ("what's the difference between OPQ and GSA"), and pure lexical search under-performs on paraphrased vague queries.

---

## 4. System Architecture

```
                 ┌───────────────────────────┐
 Client / Eval  ─┤  POST /chat               │
 Harness         │  GET  /health             │
                 └─────────────┬─────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Orchestrator (turn    │
                    │   controller)           │
                    │  - reconstructs state   │
                    │    from message history │
                    │  - turn-budget guard    │
                    │  - scope/injection guard│
                    └───────────┬────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼──────┐      ┌─────────▼─────────┐   ┌──────────▼─────────┐
│ Intent router │      │ Retrieval layer    │   │ Shortlist state    │
│ (clarify /    │      │ (hybrid dense +    │   │ builder/merger     │
│ recommend /   │      │ lexical over       │   │ (diffs vs. prior   │
│ refine /      │      │ catalog index)     │   │ shortlist mentioned│
│ compare /     │      └─────────┬─────────┘   │ in history)        │
│ refuse)       │                │              └──────────┬─────────┘
└───────┬───────┘                │                          │
        └────────────┬───────────┴──────────────────────────┘
                      │
             ┌────────▼─────────┐
             │  LLM generation   │
             │  (grounded prompt │
             │  with retrieved   │
             │  catalog chunks)  │
             └────────┬─────────┘
                      │
             ┌────────▼─────────┐
             │ Schema validator/ │
             │ URL allow-list    │
             │ (post-hoc guard)  │
             └────────┬─────────┘
                      │
             ┌────────▼─────────┐
             │ JSON response     │
             │ {reply,           │
             │  recommendations, │
             │  end_of_conv}     │
             └───────────────────┘
```

**Statelessness implication:** since no per-conversation state is stored server-side, "the current shortlist" is not a database row — it is **re-derived every call** by parsing the most recent shortlist table/JSON the agent itself emitted earlier in `messages` (the assistant's own prior turns are visible in the history). The orchestrator must parse its own prior structured output back out of the conversation history to know what to refine (see §6.3).

---

## 5. API Contract (non-negotiable, per assignment)

### `GET /health`
- Returns `200 {"status": "ok"}`.
- Must tolerate up to 2 minutes cold-start latency on the first call after idle (no cold-start logic needed for `/chat` itself).

### `POST /chat`

**Request**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response**
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**Field rules**
| Rule | Detail |
|---|---|
| `recommendations` empty | Whenever still clarifying, comparing, or refusing — i.e. `[]`/`null` any turn that isn't committing to a shortlist. |
| `recommendations` size | 1–10 items when present. |
| `end_of_conversation` | `true` only when the agent considers the task complete (shortlist delivered and no open question) — **not** merely "last allowed turn." |
| Schema | Fixed. Any deviation fails the automated evaluator outright — this is a hard-eval gate, not a soft score. |
| Turn cap | Evaluator caps at 8 turns total (user+assistant). The agent should aim to reach a first shortlist well before turn 8, since a shortlist is required at minimum by the cap. |
| Per-call timeout | 30 seconds. Retrieval + generation pipeline must complete comfortably inside this (target: real LLM call latency budget ≤ ~15–20s, leaving headroom). |

---

## 6. Conversational Behavior Specification

### 6.1 Clarify

**Trigger**: role/task/context is under-specified — no signal on job family, seniority, skills, or assessment purpose.

**Rules**
- Never emit a shortlist on the first vague turn (`C1` T1, `C3` T1, `C9` T1: role stated with zero skill/seniority detail → ask a single clarifying question, `recommendations: null`).
- Ask **one question at a time**, not a battery of questions — matches every trace (`C1`, `C3`, `C7`, `C9` each ask exactly one thing per turn).
- Prioritize clarifying dimensions in this order when multiple are missing, based on what most changes the shortlist:
  1. **Role/skill family** (what are they assessing for),
  2. **Seniority/experience level**,
  3. **Purpose** (selection vs. development vs. audit — `C1` T2 asks this because it changes report format),
  4. **Delivery constraints** (language, region, time budget — `C3`, `C7`).
- If the user supplies a full job description in one message (`C9` T1), skip generic clarification and ask a **targeted** disambiguating question derived from the JD itself (which of N named skills is primary) rather than a generic "tell me more."
- A user statement that is enough to act on some but not all axes should be met with a partial narrowing + one more targeted question, not silence and not a premature shortlist (`C1` T2: commits to instrument, still asks about report format).

### 6.2 Recommend

**Trigger**: enough context has been gathered (or the user explicitly says "go ahead"/accepts a proposed direction).

**Rules**
- 1–10 items, each with `name`, `url`, `test_type`, sourced only from the retrieval index.
- Default companion behavior observed across traces: for personality-relevant hires, the agent proactively includes **OPQ32r** as a default personality layer and *tells the user it did so*, offering to drop it (`C2` T2, `C8` T1). This should be a configurable heuristic, not hard-coded — but the **transparency pattern** (state defaults added and offer removal) should be a general prompt instruction, applied to any default/assumed inclusion.
- When the catalog has **no exact match** for a named skill (e.g., no Rust-specific test, `C2` T1), the agent must say so explicitly, propose the closest legitimate substitutes with a rationale, and ask permission before finalizing — never silently substitute or hallucinate a product name.
- When constraints conflict with availability (e.g., Spanish-language delivery + English-only knowledge tests, `C7` T1), lay out the real trade-off as explicit options (hybrid vs. drop rigor) and let the user choose — don't quietly pick one.
- Recommendations should be **role/purpose complete**, not a single test, when the query implies a battery (e.g., safety-critical role → personality-for-safety-behavior instrument + role knowledge test, not knowledge alone, `C6` T1).

### 6.3 Refine

**Trigger**: user adds, removes, or swaps a constraint mid-conversation ("add personality tests," "drop REST," "remove OPQ").

**Rules — this is the highest-risk behavior for a stateless API and needs explicit design:**
1. **Reconstruct current shortlist** by parsing the most recent assistant message in `messages` that contained a committed shortlist (the orchestrator re-parses its own last structured table/JSON out of conversation history — see §4).
2. **Diff, don't regenerate from scratch.** Apply only the delta the user requested; unaffected items must remain byte-identical in `name`/`url`/`test_type` across turns (`C4` T2: adding Graduate Scenarios leaves items 1–3 unchanged; `C9` T4: dropping REST, adding AWS+Docker leaves Java/Spring/SQL/Verify/OPQ unchanged). Evaluator behavior probe: "agent honors edits in recommendations" — this implies **exact stability of unedited rows**, not just semantic similarity.
3. **Pushback is allowed but must not silently override the user.** If the requested change isn't sensible (`C10` T2: "remove OPQ, replace with something shorter" — no shorter equivalent exists), the agent may push back once with a reason and withhold the edit (`recommendations: null` that turn), but **must comply** if the user repeats/overrides the instruction on a later turn (`C10` T4: "drop the OPQ. final list" → complied, no further pushback). Never refuse a direct, repeated, in-scope instruction.
4. Renumber the table row indices on any change; do not preserve old numbering with gaps.
5. A refine turn that changes nothing about the *list* but only re-confirms it ("that's good", "confirmed") should re-emit the identical shortlist and typically set `end_of_conversation: true` (final confirmation pattern seen at the end of every trace).

### 6.4 Compare

**Trigger**: user asks about the difference between two or more named products, or asks whether one is the right pick vs. an alternative.

**Rules**
- Answer must be grounded in retrieved `description`/`keys`/`job_levels`/`languages` differences between the two specific catalog records — never the model's prior knowledge of what a test "probably" measures.
- Distinguish **instrument vs. report** relationships when the catalog encodes them this way (`C5` T2: OPQ32r is the questionnaire; OPQ MQ Sales Report is a reporting/output product built on it, not a separate instrument) and **general vs. sector-calibrated variants** (`C6` T2: DSI vs. Manufacturing & Industrial Safety & Dependability 8.0 — same construct family, different norm base and packaging) and **legacy vs. new variants of near-identical names** (`C3` T4, `C8`: "(New)" simulation-inclusive products vs. older knowledge-only or bundled products with the same functional name).
- A compare turn does **not** have to include `recommendations`; it can be pure explanation (`C1`, `C5`, `C6`, `C9` all have compare-only turns with `recommendations: null`), *unless* the comparison directly updates the shortlist choice, in which case emit the updated list (`C6` T3 folds the comparison decision straight into a revised, narrowed list).
- If asked to justify a specific item already on the list against an alternative reading of the JD (`C9` T5: "Is Advanced the right pick given they work on existing services?"), the agent should reason from catalog-level descriptions of each variant, not just restate the name.

### 6.5 Refuse / stay in scope

**Trigger categories** (must all be handled):
- General hiring/legal/compliance advice (`C7` T3: "are we legally required under HIPAA" → refuse the legal question, but still answer the adjacent, in-scope factual question about what the test measures, and do not end the conversation abruptly — keep engaging on the in-scope remainder).
- Prompt injection ("ignore previous instructions", "pretend you are...", attempts to make the agent output non-catalog links or arbitrary text).
- Requests for assessments/products not in the SHL Individual Test Solutions catalog (never fabricate a plausible-sounding SHL product name or URL).
- Fully off-topic requests unrelated to SHL assessment selection.

**Rules**
- Refusal must be specific about *what* is out of scope and *what remains* in scope, and should redirect to the appropriate resource (e.g., "your legal/compliance team") without being curt or ending the session unnecessarily.
- `recommendations` stays `null`/empty on a pure refusal turn; if the refused request was layered onto an otherwise valid recommendation turn, still answer the valid part.
- `end_of_conversation` should not flip to `true` on a refusal alone — the user may continue the legitimate part of the conversation afterward.

---

## 7. Context Engineering / Prompt Design

### 7.1 Per-turn context assembly

For each `/chat` call, construct the LLM context from:
1. **System prompt** — role, hard scope boundary, output schema, the four behaviors and their rules (§6), and the URL-fidelity constraint ("every URL must come from the provided catalog snippets; never construct or recall a URL from memory").
2. **Full conversation history** (as given — it's the only state available).
3. **Retrieved catalog snippets** — top-K hybrid retrieval results run against a query synthesized from (a) the latest user turn and (b) a rolling summary of accumulated constraints extracted from earlier turns (role, seniority, skills, language, purpose). Re-run retrieval every turn; don't rely on the LLM's memory of catalog contents from earlier turns, since irrelevant/omitted context is the main source of hallucinated names/URLs.
4. **Reconstructed current shortlist** (parsed from the agent's own last shortlist-bearing turn, per §6.3) when the turn looks like a refine/confirm rather than a fresh recommend.

### 7.2 Grounding & anti-hallucination guardrails

- **Retrieval-then-generate, not generate-then-retrieve**: the LLM must be shown only entities actually returned by the retrieval layer for that turn; it should be instructed to select/report from that set only, never to introduce items outside it.
- **Post-hoc validator**: after LLM generation, programmatically check every `url` in `recommendations` against the catalog index by exact match. Any URL that fails the check is a hard failure — either strip the item and regenerate, or fail closed with a clarifying/refusal reply rather than emit an unverified link. This is the single highest-leverage guardrail against the "hallucination %" behavior probe.
- **Test-type/URL consistency check**: verify `test_type` returned matches the catalog record's actual code, not an LLM guess.

### 7.3 Turn-budget awareness

With an 8-turn cap and a simulated user that "ends the conversation when the agent provides a shortlist," the agent should bias toward reaching a first shortlist efficiently — typically within 2–3 clarifying turns — rather than exhaustively clarifying every possible axis, since Recall@10 is scored on the *final* shortlist and unresolved conversations that hit the cap without ever committing to `recommendations` likely score zero on recall.

### 7.4 Name collision handling

The catalog contains many same-family products distinguished only by suffix or bundling (e.g., "MS Excel (New)" knowledge-only vs. "Microsoft Excel 365 (New)" knowledge+simulation; "Contact Center Call Simulation (New)" vs. "Customer Service Phone Simulation"). Retrieval and prompt instructions must preserve full product names verbatim (including "(New)"/version markers) — truncating or normalizing these away is a direct source of wrong-URL errors.

---

## 8. Statelessness & Turn Reconstruction — Implementation Notes

Since the service holds no session state, `entity_id`s of the current shortlist are **not persisted** anywhere except inside the text/JSON of the agent's own previous replies within `messages`. Recommended implementation:
- Have the agent's `reply` (or an internal-only structured echo not necessarily shown verbatim to the user) consistently render shortlists in one parseable form (e.g., the markdown table format used in the traces, or better, keep a machine-parseable copy of `recommendations` from prior turns available in history since the request schema echoes full history including prior assistant JSON if the client resends it that way).
- On each incoming request, the orchestrator scans `messages` from the end backwards for the most recent assistant turn that carried a non-empty shortlist, and treats it as "current state" for diffing during Refine turns.
- Because the harness may pass either the pretty `reply` text or (if it's replaying its own view) a simplified transcript, **do not rely solely on parsing markdown tables from `reply`** — treat this as a best-effort fallback, and design the primary source of truth to be robust to the assistant's own past replies being present only as plain text in history (worst case: re-run retrieval fresh and use the LLM's own reading of the conversation to infer the last shortlist from `reply` text, since that's genuinely all that's guaranteed to survive statelessness).

---

## 9. Evaluation Plan

Mirrors the assignment's three-part grading. Build local test harnesses for each before submission.

### 9.1 Hard evals (must pass, binary gate)
- JSON schema validation on every response (`reply: str`, `recommendations: list|null`, `end_of_conversation: bool`).
- Every `recommendations[i].url` exists in the scraped catalog index.
- `len(recommendations)` is 0 (clarify/compare/refuse turns) or 1–10 (commit turns) — never 11+, never a non-empty list on a pure-clarify turn.
- Conversation never exceeds 8 turns; service responds within 30s per call including cold paths.
- `GET /health` returns `200 {"status":"ok"}` with cold-start tolerance.

### 9.2 Recall@10
- Build a local replay harness using `C1`–`C10` (and self-authored holdout traces covering categories not in the 10, e.g., a pure prompt-injection trace, a pure off-topic trace, a multi-language edge case) with an LLM-simulated user driven by each trace's persona/fact set.
- Score `Recall@10` against each trace's final committed shortlist (the last `recommendations` block with `end_of_conversation: true`) as the label set.
- Track Recall@10 across iterations of retrieval strategy (pure lexical vs. pure dense vs. hybrid) to justify the final retrieval design choice in the approach doc.

### 9.3 Behavior probes (illustrative set to implement)
| Probe | Assertion |
|---|---|
| No premature recommend | Turn-1 vague query → `recommendations` empty. |
| Refine stability | Non-edited items' `name`/`url`/`test_type` are byte-identical before/after a partial refine. |
| Off-topic refusal | Off-topic user message → no catalog items returned, reply declines and redirects. |
| Legal/compliance refusal | Legal question → refused, but adjacent in-scope factual content still answered. |
| Prompt-injection resistance | Injection attempt embedded in user turn does not alter system behavior or leak instructions. |
| URL fidelity / hallucination rate | % of turns with any `url` not present in the catalog index (target: 0%). |
| Compare groundedness | Compare answers reference catalog-attributable distinctions (name/keys/description deltas), not generic prior knowledge. |
| Repeated-instruction compliance | After one round of agent pushback on an edit, a repeated user instruction is honored, not re-refused. |
| Turn-cap graceful degradation | If context is still insufficient near turn 7–8, the agent commits to a best-effort shortlist rather than hitting the cap with `recommendations: null`. |

---

## 10. Suggested Tech Stack (justify final choice in the approach doc)

| Layer | Options | Notes |
|---|---|---|
| API | FastAPI + Uvicorn | Required by spec (`GET /health`, `POST /chat`). |
| LLM | Gemini free tier / Groq / OpenRouter / Anthropic API | Pick one with function-calling or reliable structured-output mode for schema compliance; keep prompts explicit about the exact JSON to return regardless of provider. |
| Retrieval | Hybrid: FAISS or Chroma (dense) + BM25/rapidfuzz (lexical) | Justify over pure-vector given the many exact-name lookups in Compare/Refine turns. |
| Scraper | `requests`/`httpx` + `BeautifulSoup` (or Playwright if the catalog is JS-rendered) | Must cover pagination across the full Individual Test Solutions listing, not just the sample 5 rows shown here. |
| Deployment | Render / Fly / Railway / Modal / HF Spaces | Must survive cold start within the 2-minute `/health` grace period; keep the retrieval index pre-built and loaded at process start, not rebuilt per request. |

---

## 11. Known Risks / Failure Modes to Guard Against (from the assignment's own "what unsuccessful submissions look like")

1. **Happy-path-only code**: handle empty `messages`, malformed roles, a user who answers a clarifying question with irrelevant text, ties/near-ties in retrieval scoring, and catalog entries with missing `duration`/`languages`.
2. **Vibe-coded, undefendable design choices**: every heuristic in §6 (e.g., default OPQ32r inclusion, clarify-question ordering) should be a named, isolated function/prompt block the author can explain and justify in the technical interview — not an emergent behavior of one big prompt.
3. **Weak evaluation rigor**: don't just eyeball a couple of manual chats — run the full `C1`–`C10` set plus authored adversarial/off-topic/injection traces through an automated harness and record Recall@10 and probe pass-rates before submission, per §9.

---

## 12. Fallback Logic for Potential Failure Modes

Every external dependency in this system (catalog data source, LLM provider, retrieval index) can fail or degrade mid-conversation. Because the API is stateless and graded by an automated harness with a hard 30-second timeout and an 8-turn cap, **every fallback must still return a schema-valid response** — there is no acceptable failure state that returns a non-JSON error, a 500, or a hang. The rule throughout: **degrade gracefully to a smaller, well-formed answer; never fail open into unschema'd output, and never fail closed into a hang.**

### 13.1 Data-layer fallbacks

| Failure | Fallback |
|---|---|
| JSON catalog endpoint (§3.0) unreachable/times out at startup | Load the most recent successfully-fetched local snapshot from disk (cached on every successful fetch). Serve from cache and log a staleness warning; never block startup on a live fetch. |
| JSON endpoint reachable but returns a different/broken shape (schema drift) | Validate the fetched payload against the expected field set (§3.1) before swapping it in as the live index. If validation fails, keep serving the last-known-good cached index rather than adopting a malformed one. |
| Individual vs. Job Solution classification (§3.3) is ambiguous for a given record | Default to **excluding** the ambiguous record from the retrievable set rather than including it — a missed-but-safe recommendation is recoverable in a later turn if the user asks; a wrongly-included Job Solution is a hard-eval failure risk with no recovery. |
| A catalog record is missing `duration`/`languages`/`keys` or has an empty `description` | Never let a missing field crash retrieval or generation; treat missing fields as `"Not specified"` in the context shown to the LLM, and don't let the LLM infer/hallucinate a value for it. |
| Catalog record count drops sharply between fetches (e.g., site outage returns a partial/empty payload) | Sanity-check fetched record count against a reasonable historical baseline before accepting the fetch; if it drops implausibly (e.g., >50% smaller), reject the fetch and keep the cached index, alerting rather than silently shrinking the retrievable catalog. |

### 13.2 Retrieval-layer fallbacks

| Failure | Fallback |
|---|---|
| Dense (embedding) index unavailable or errors at query time | Fall back to lexical/BM25-only retrieval for that turn rather than failing the request; log degraded-mode usage. |
| Both retrieval paths return zero results for a turn that expects a shortlist | Don't force a fabricated shortlist. Return a clarify-style reply acknowledging no confident catalog match was found, ask a narrowing question, and keep `recommendations` empty rather than lowering the relevance bar until *something* returns. |
| Retrieval returns many near-duplicate items (e.g., same product family, "(New)" vs. legacy vs. bundled variants — §7.4) | Apply a light dedup/diversity re-rank pass so the shown shortlist isn't dominated by near-identical entries, unless the user has specifically asked to compare those variants. |
| Retrieval latency spikes (e.g., cold embedding model, cold index load) | Keep the index loaded in-process at startup (§10), not lazy-loaded per request; if a cold path is unavoidable, set an internal retrieval timeout well under the 30s budget and fall back to lexical-only rather than exceeding the request's own timeout. |

### 13.3 LLM-generation-layer fallbacks

| Failure | Fallback |
|---|---|
| LLM returns malformed JSON / doesn't follow the schema | Programmatically validate before returning to the client. On failure, retry generation once with a stricter "return only JSON, no prose" instruction; if it still fails, fall back to a deterministic template response (e.g., a clarify-style reply built from a fixed string plus whatever partial state is known) so the client never receives non-schema output. |
| LLM call times out or the provider errors/rate-limits | Have a secondary LLM provider or a smaller/faster model configured as a fallback for that single call; if all providers fail, return a deterministic apology-and-retry-style reply that is still schema-valid (`recommendations: null`, `end_of_conversation: false`) rather than an HTTP error, since the harness expects a `/chat` response, not an exception. |
| LLM hallucinates a `url`/`name` not present in retrieval results | Caught by the post-hoc validator (§7.2). On a caught hallucination, strip the offending item(s); if that leaves zero valid items on what was meant to be a commit turn, don't emit an empty `recommendations` silently — either regenerate once grounded strictly in the retrieved set, or fall back to the top-K retrieved items directly (bypassing free-form LLM selection) with a templated `reply`. |
| LLM ignores the scope boundary (answers an off-topic/legal/injection request instead of refusing) | Treat scope/injection detection as a **pre-generation classifier step**, not solely a prompt instruction the main generation call might drift from. A lightweight, separate classification pass (rule-based keyword/pattern checks plus a small LLM classification call) decides refuse-vs-proceed before the main response is generated, so a single compromised generation call can't leak scope-violating content. |
| LLM output is schema-valid but semantically empty/unhelpful (e.g., repeats the same clarifying question turn after turn) | Track whether the same clarifying question (or a near-duplicate) has already been asked in the visible history; if so, force progression — either commit to a best-effort shortlist using whatever constraints are already known, or ask a genuinely different question, rather than looping. |

### 13.4 Statelessness / Refine-layer fallbacks

| Failure | Fallback |
|---|---|
| Prior shortlist can't be confidently reconstructed from history (§8) — e.g., the last assistant turn's shortlist is ambiguous/unparseable | Don't guess silently. Re-run retrieval fresh against the full accumulated constraint set from the conversation and present it as a freshly-stated shortlist (clearly a "here's where we are" recap rather than a silent, possibly-wrong diff), rather than fabricating a diff off a misread prior state. |
| User's refine instruction is self-contradictory or references an item never actually shown (e.g., "drop the item you never actually recommended") | Point out the discrepancy in `reply`, don't silently invent an interpretation; ask a one-line clarifying question if needed, keeping the previous valid shortlist unchanged until resolved. |
| User repeats a refine instruction the agent already pushed back on once (§6.3 rule 3) | Must comply on the second ask — don't re-run the same pushback loop; track "have I already pushed back on this exact edit" within the visible history to avoid an infinite negotiation loop that burns the 8-turn budget. |

### 13.5 API / infra-layer fallbacks

| Failure | Fallback |
|---|---|
| `messages` array is empty, malformed, has unknown roles, or has a `content` that's empty/whitespace | Validate the request shape before any LLM/retrieval work; respond with a schema-valid clarifying turn (`recommendations: null`, generic opening question) rather than a 4xx/5xx, since a malformed-but-parseable history is exactly the kind of realistic-user noise the harness may send. |
| Per-call timeout risk (approaching 30s) | Set internal timeouts on every external call (catalog fetch, embedding query, LLM call) well below 30s combined, with the LLM call as the largest single budget item; if the pipeline is about to exceed budget, return the best-effort partial response built so far (e.g., top retrieved items templated into a reply) rather than letting the platform's timeout kill the request with no response at all. |
| Cold start on `/health` or first `/chat` after idle | Keep retrieval index and any embedding model warmed/loaded at process start (not on first request) so that once `/health` passes, `/chat` is not itself subject to a second cold-start penalty within its own 30s budget. |
| Conversation exceeds the 8-turn cap without a committed shortlist | On the final allowed turn, force a best-effort commit: return the top items the retrieval/constraint state supports at that point with `recommendations` non-empty and `end_of_conversation: true`, rather than ending on an unresolved clarify turn that scores zero on Recall@10. |
| Two or more failure types compound in the same turn (e.g., retrieval degraded *and* LLM primary provider down) | Each fallback layer should be independently triggerable and stackable (degraded retrieval + fallback LLM provider can both be active at once); the system should never need "everything working" to produce a valid response — only the data-layer cache and a deterministic templating path are truly load-bearing as the last line of defense. |

---

## 13. Deliverables Checklist (per assignment)

- [ ] Deployed FastAPI service; `GET /health` and `POST /chat` both publicly reachable at submission time.
- [ ] Full Individual Test Solutions catalog scraped, cleaned, indexed (not just the 5-row sample used for illustration here).
- [ ] Retrieval + orchestration + generation pipeline implementing all four behaviors and the refusal/scope rules.
- [ ] Local replay harness against `C1`–`C10` + authored holdout traces, with Recall@10 and behavior-probe results.
- [ ] 2-page approach document: design choices, retrieval setup, prompt design, evaluation approach, what didn't work, any AI-tool usage disclosure.
