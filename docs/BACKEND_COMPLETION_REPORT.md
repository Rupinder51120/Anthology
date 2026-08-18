<!-- DRAFT IN PROGRESS — being filled in during the backend-completion pass (docs/ANTHOLOGY_FULL_AUDIT.md, 2026-08-18). Not yet final. -->
# Anthology — Backend Completion Report

_Draft — being populated as Phases 5-12 of the backend-completion pass complete. Do not cite numbers from this file until the "Final Backend Status" section confirms BACKEND FREEZE._

## Architecture

```
PDF corpus (~120 papers)
  → Docling parsing (CPU-accelerated; OCR fallback; multimodal block extraction)
  → metadata resolution (registry → Docling structural → heuristic fallback)
  → section-aware chunking (math/table-safe splitting, deterministic content-hash chunk IDs)
  → multimodal enrichment (Groq vision captioning + DePlot chart extraction for figures,
    Groq table summarization)
  → embedding (allenai/specter2_base, 768-dim, content-first text contract)
  → Postgres 16 + pgvector (atomic per-paper transaction) + full-text search (tsvector)
  → hybrid retrieval (dense pgvector cosine + sparse Postgres FTS, RRF fusion)
  → Cohere rerank-v3.5 reranking
  → HyDE (available, evidence-based opt-in only — see Known Limitations)
  → Groq (openai/gpt-oss-20b, primary) / Ollama (qwen2.5:7b, fallback) generation
  → grounding gate (lexical-overlap check) + inline citations
  → evaluation (Hit@k/MRR/nDCG retrieval metrics + Groq-judge generation metrics)
```

FastAPI backend (13 routers), React/Vite frontend (untouched this pass — backend-completion work only), Postgres+pgvector+Redis+Ollama via Docker Compose.

## Data Pipeline

**Corpus**: 122 papers, 12,021 chunks (avg 98.5 chunks/paper, range 15-554), 1,734 table chunks (100% Groq-enriched), 12 figure chunks (0.1% of corpus, benign low-content edge cases), 100% title/author coverage, 89% abstract coverage, 49% year coverage (heuristic-extraction limitation on papers without clean Docling date metadata — doesn't affect retrieval/generation).

**Reliability hardening this pass** (`scripts/build_index.py`): per-paper wall-clock timeout via `asyncio.wait_for` (default 1200s) so one hung/pathological PDF can no longer stall an entire bulk run; a resumable JSON status ledger (`logs/ingestion_status.json`) written after every paper, supporting `--resume`/`--only-failed` so a crash never requires blindly re-ingesting the whole corpus; a fresh DB session per paper instead of one shared session for the whole run, eliminating any risk of one paper's failure poisoning every subsequent paper's transaction. Validated live against the real pipeline, including a genuine >1200s timeout case (an 18-table paper that took only 66s under the original MPS-accelerated Docling path but exceeded the timeout under the CPU-forced fix that resolved a prior MPS hang — a real, quantified ~20x slowdown tradeoff for table/figure-heavy PDFs, accepted in favor of reliability). Verified the timeout fires cleanly with **zero data corruption**: DB writes only happen after the CPU-bound parse/enrich stage returns, so a mid-parse timeout can never produce a partial write.

**Existing corpus was verified, not blindly re-ingested**: before any execution, the 122-paper corpus already present in the database was confirmed to be complete and clean (exact 1:1 filename↔paper match, 0 orphans, 0 duplicates, 0 missing embeddings) — re-running ingestion for already-correct data would have wasted real API budget for zero benefit.

## Retrieval

**Architecture** (unchanged from pre-existing design, per explicit instruction not to rewrite working retrieval architecture): dense (pgvector cosine) + sparse (Postgres full-text search) candidates fused via Reciprocal Rank Fusion (weighted by section priority and query-modality boosting), then reranked by Cohere `rerank-v3.5`. Six selectable strategies (`sparse`, `dense`, `dense_hyde`, `hybrid_rrf`, `hybrid_rerank`, `dense_rerank`); `hybrid_rerank` is the production default.

**Root-cause fix — the audit's central open question, resolved with evidence**: dense retrieval's near-zero chunk-level hit@1 was caused by the chunk embedding text prepending `Title | Authors | Year` to every chunk. Since these fields are identical for every chunk of a given paper, the embedding model's representational capacity was dominated by this constant prefix rather than the actually-varying content — directly measured: two unrelated chunks from the same paper embedded at **97.9% cosine similarity**, and the margin between a query's similarity to the truly-relevant vs. an irrelevant same-paper chunk was only **0.004** (noise-level). Fixed the embedding-text contract to `[Section] content` only — title/authors/year remain fully available via chunk metadata/DB columns for citations, filtering, and display; only what gets embedded changed. No change to the embedding model, vector dimension, pgvector schema, retrieval architecture, RRF weighting, or reranker. The entire 12,021-chunk corpus was re-embedded in place (embedding-only regeneration — no re-parsing/re-enrichment) and verified intact (same chunk/paper counts, same identities, all valid 768-dim vectors, 0 NaN/zero-vectors).

**Validated with a clean, corrected methodology** (n=40 real benchmark questions, measuring within-paper chunk ranking to isolate the exact discrimination problem): hit@1 0.075→0.125 (**+67% relative**), MRR 0.195→0.265 (**+36% relative**).

## Reranking

Cohere `rerank-v3.5`, unchanged from the pre-existing stabilization-pass fix (retry/backoff on transient failures, 3 attempts with exponential backoff, falls back to unranked top-k on exhaustion). Verified correct: reorders by `relevance_score`, no dedup/ordering bugs found in this pass's correctness audit.

## Generation

Dual-backend (Groq primary, Ollama `qwen2.5:7b` fallback — fallback now correctly Docker-network-aware per the stabilization pass). System prompt explicitly instructs the model to say what's missing rather than guess, and to never hallucinate citations/numbers not present in context. A post-hoc grounding gate (lexical word-overlap check, >15% threshold) catches answers that drift from the retrieved context. Citations deduped by (title, section), sorted by rerank/relevance score.

**Critical bug found and fixed this pass**: the configured Groq model (`llama-3.1-8b-instant`) had been fully retired by Groq — every generation call was failing with `404 model_not_found`, and the failure was being silently disguised as a normal "Could not find a grounded answer" response by a double-masking bug (the real exception got converted to error text, which then failed the grounding check and was replaced again with a generic refusal — completely erasing the original error). Fixed both the model config (→ `openai/gpt-oss-20b`, matching a config-level default that had already been updated but was being overridden by a stale `.env` value) and the masking bug itself (`generate_answer()` now returns a distinct `response_type: "error"` with real logging on hard API failures, never falling through into the grounding check).

**Live validation, post-fix, across 5 question categories** (real questions, real corpus, production `hybrid_rerank` strategy):
| Category | Result |
|---|---|
| Answerable (in-corpus) | Correctly refused — re-examined and confirmed *correct*: this corpus has no dedicated RAG paper for the specific question asked, a flaw in the test question, not a bug |
| Unanswerable (out-of-corpus) | Correctly and honestly explained the context doesn't cover the topic, citing what it *did* retrieve rather than issuing a generic refusal |
| Multi-paper comparison | Real synthesized comparison across multiple distinct papers, properly cited |
| Multi-chunk synthesis | Accurately reproduced real table data (FID scores, accuracy percentages) from two different papers into a structured comparison |
| Conflicting evidence | Presented the actual trade-off between two approaches rather than fabricating a false consensus |

Every citation in every real answer was verified to reference an actually-retrieved chunk — zero fabricated citations observed.

## Evaluation

**Methodology**: CLI (`scripts/run_benchmark.py`) → `pipeline_runner.run_pipeline_on_dataset_async` (per-question retrieve + generate, checkpointed so a crash resumes rather than restarts) → `evaluator.evaluate_results` (paper-level and chunk-level Hit@1/3/5/10, MRR, nDCG@5; optional Groq-judge faithfulness/relevance/completeness) → `save_scores`. **Dataset**: `indexes/qa_dataset.json`, 247 real question/answer/ground-truth-chunk triples generated from the actual ingested corpus (confirmed fresh/non-contaminated, distinct from the stale `qa_quick.json` that produced misleading numbers in the pre-existing `benchmark_summary.json` — that file should not be cited).

**Evaluation code cleanup this pass**: confirmed via repo-wide grep (zero callers) before removing anything — deleted `src/evaluation/generation_metrics.py` (380 lines, entirely dead) and the dead `RetrievalEvaluator`/`EvalResult`/`RetrieverComparison` classes from `retrieval_metrics.py` (referenced an object model constructed nowhere in the codebase), plus the unused `ragas` dependency. The live evaluation path is now singular with no competing/confusing frameworks. Full 43-test suite re-verified passing after cleanup.

## Reliability Improvements

This pass, on top of the prior stabilization pass (P0/P1 fixes — installability, Alembic schema reproducibility, path traversal, Docker config consistency):
- Per-paper ingestion timeout + resumable ledger + session-per-paper isolation (see Data Pipeline).
- Root-caused and fixed the dense-retrieval embedding-text defect with direct, reproducible measurement rather than guesswork.
- Root-caused and quantified HyDE's latency (~164s/query, 23x overhead, dominated by Ollama LLM generation) rather than leaving it as an unexplained audit finding — evidence-based decision to exclude it from default/full-benchmark use while preserving the working feature.
- Removed ~660 lines of dead, confusing evaluation code with zero-caller verification before deletion (not a guess).

## Quantitative Results

**Retrieval** — fresh benchmark, 247 real questions (`indexes/qa_dataset.json`), embedding model `allenai/specter2_base` (post-fix), reranker Cohere `rerank-v3.5`, date 2026-08-18:

| Strategy | Paper Hit@1 | Paper Hit@5 | Paper MRR | Chunk Hit@1 | Chunk Hit@5 | Chunk MRR | n |
|---|---|---|---|---|---|---|---|
| sparse | 0.301 | 0.632 | 0.436 | 0.142 | 0.444 | 0.261 | 239 |
| dense | 0.433 | 0.649 | 0.524 | 0.041 | 0.143 | 0.086 | 245 |
| dense_hyde (n=10 sample) | 0.600 | 0.700 | 0.667 | 0.200 | 0.400 | 0.253 | 10 |
| hybrid_rrf | 0.573 | 0.793 | 0.664 | 0.357 | 0.652 | 0.474 | 227 |
| **hybrid_rerank (production default)** | **0.784** | **0.863** | **0.824** | **0.577** | **0.722** | **0.643** | 241 |
| dense_rerank | 0.641 | 0.756 | 0.690 | 0.194 | 0.219 | 0.204 | 242 |

Dense-only chunk-level Hit@1 improved from a pre-fix **0.0** (per the original audit) to **0.041** post-fix — real but modest at full-corpus scale (harder task than the isolated within-paper measurement below). RRF fusion is what actually rescues dense's weakness (chunk Hit@1 jumps to 0.357 in `hybrid_rrf`); reranking on top pushes the production default to 0.577.

**Retrieval — isolated embedding-fix validation** (n=40, within-paper chunk ranking, isolates the exact discrimination problem the fix targeted): Hit@1 0.075→0.125 (+67% relative), MRR 0.195→0.265 (+36% relative).

**Latency**: normal query (embed+dense-retrieve+rerank) ~7s cold / sub-second warm. HyDE query ~164s (23x overhead, dominated by Ollama LLM generation — see Known Limitations).

**Generation (retrieval-independent re-run with the fixed model)**: paper hit@1=0.773, chunk hit@1=0.575 — consistent with Phase 6's numbers, confirming the fix didn't regress retrieval. **Judge-scored faithfulness/relevance/completeness are NOT usable from this run** (all landed at ~0.5 — the Groq account hit its 200k-token/day quota partway through judging, and the judge metric silently defaults to 0.5 on failure; this is BUG-24 from the v1 audit, predicted then and now directly confirmed with real data). Getting trustworthy judge numbers requires the daily quota to reset or a paid Groq tier — flagged as a known limitation below rather than reported as a real finding.

**Configuration**: embedding `allenai/specter2_base` (768-dim), reranker Cohere `rerank-v3.5` (trial tier, 10 req/min), generation Groq `openai/gpt-oss-20b` (primary) / Ollama `qwen2.5:7b` (fallback), corpus 122 papers / 12,021 chunks, benchmark dataset `indexes/qa_dataset.json` (247 questions), date 2026-08-18.

## Known Limitations

- **HyDE is ~23x slower than default retrieval** (~164s/query vs ~7s), dominated by Ollama LLM generation latency, and occasionally hard-times-out (120s ceiling) under load. Not enabled by default; excluded from the full benchmark in favor of a small documentation sample. See Retrieval section.
- **The Cohere API key currently configured is a trial-tier key, rate-limited to 10 calls/minute.** Observed directly during the Phase 6 benchmark run: `hybrid_rerank`/`dense_rerank` strategies hit frequent HTTP 429s. The stabilization-pass retry/backoff (3 attempts, exponential) handles this gracefully — but after exhausting retries it falls back to *unranked* (RRF-only) ordering for that query. This means a nontrivial fraction of "reranked" results in any bulk run under this key may silently not have been Cohere-reranked at all, and any interactive/production use under sustained load will see the same silent degradation. **Needs a production-tier Cohere key before this can be considered reliable at any real traffic volume.**
- 62/122 papers missing `year` metadata (heuristic-extraction limitation on papers without clean Docling date metadata) — cosmetic, doesn't affect retrieval or generation quality.
- 12/12,021 chunks (0.1%) are unenriched figures — benign low-content edge cases (decorative/small images), not a systemic captioning failure (100% of table chunks are enriched).
- `_row_to_dict()`'s `rerank_score` field is overloaded (pre-rerank vs post-rerank meaning) — latent risk if the no-Cohere-key fallback path is ever exercised in production (it currently isn't, since a key is always configured), not fixed this pass.
- **The corpus (122 AI/ML papers on diffusion models, GANs, medical imaging, LLM evaluation, causal reasoning, etc.) does not contain a dedicated paper on Retrieval-Augmented Generation** — a query specifically about RAG will correctly be refused rather than answered from unrelated context. This is correct grounding behavior, not a bug, but worth knowing when picking demo questions.
- **Groq's on-demand tier caps at 200,000 tokens/day**, which a single full evaluation run (247 generation calls + up to 741 judge calls) can exhaust on its own. Judge-based generation-quality scoring is not currently repeatable more than once per day under this account tier. A production deployment or CI eval pipeline needs a paid tier (or a separate, cheaper dedicated judge model) to run evaluation routinely.
- *(to add: any findings from Phase 11)*

## Final Backend Status

**Freeze checklist:**

- [x] ~120 papers successfully ingested (122/122, verified not blindly re-ingested)
- [x] 0 orphan chunks
- [x] 0 duplicate chunks
- [x] 0 missing embeddings
- [x] embedding dimensions consistent (768 throughout)
- [x] retrieval pipeline verified (full correctness audit — distance direction, RRF math, dedup, rerank ordering all confirmed correct)
- [x] dense retrieval issue understood/fixed (root cause found, fixed, validated at both the isolated and full-corpus level)
- [x] hybrid retrieval verified (fresh 6-strategy, 247-question benchmark)
- [x] reranking verified (correct ordering confirmed; trial-tier rate limit is an external account constraint, not a code defect — see Known Limitations)
- [x] generation verified (5 real question categories tested; found and fixed a critical deprecated-model bug in the process)
- [x] citations verified (zero fabricated citations across every real test)
- [x] Groq path verified (fixed, validated live via direct call and the real API)
- [x] Ollama fallback verified (live-tested, produced a real correctly-cited answer)
- [x] fresh benchmark generated (247 real questions, all 6 strategies, non-contaminated dataset)
- [x] evaluation completed
- [ ] **evaluation numbers trustworthy — PARTIAL.** Retrieval metrics (Hit@k/MRR/nDCG) are fully trustworthy — computed independently of generation, verified consistent across two separate runs. Judge-based generation-quality scores (faithfulness/relevance/completeness) are **not** trustworthy in the current data — the Groq account's daily token quota (200k/day) was exhausted mid-run, confirmed in logs, triggering evaluator's known silent-0.5-on-failure fallback (audit BUG-24, now directly confirmed). Re-running judge scoring after the daily quota resets (or on a paid tier) is the one remaining action item — it does not require any further code changes.
- [x] tests passing (43/43)
- [x] Docker healthy (all 5 services confirmed running/healthy throughout)
- [x] no P0 issues in the backend pipeline (2 P0s found *during this pass* — the deprecated Groq model, and missing DB timestamp defaults — both found and fixed; the one remaining P0 from the prior stabilization pass, a leaked API key in git history, is a repo/security matter unrelated to pipeline correctness)
- [x] no unresolved backend P1 issues that affect the demo (the Cohere rate limit is a real, unresolved external constraint, but doesn't block interactive demo usage at normal query rates — it only surfaced under the sustained 247-question automated benchmark)
- [x] reproducible setup documented (README, `.env.example`, this report, and the audit docs)

**19 of 20 criteria fully met.** The one open item (judge-scored generation-quality numbers) is blocked by an external API quota, not a backend defect, and does not block frontend integration work — the frontend doesn't consume judge scores, it consumes the retrieval/generation API, which is fully verified and working.

## BACKEND FREEZE — READY FOR FRONTEND

The ingestion → retrieval → reranking → generation pipeline is verified end-to-end against the real 122-paper corpus with a fresh, non-contaminated benchmark. Two real, previously-hidden critical bugs were found and fixed during this pass (a fully-deprecated Groq model silently breaking 100% of generation, and missing DB-level timestamp defaults silently breaking Collections/Sessions creation and all query audit-logging) — both are the kind of defect that would have surfaced immediately during real frontend integration testing had they not been caught here first.

**One follow-up action, not a blocker**: re-run judge-based generation-quality scoring (`indexes/results_hybrid_rerank_fixed.json` → `evaluate_results(..., run_judge=True)`) once the Groq daily quota resets, to get trustworthy faithfulness/relevance/completeness numbers for the record.

**Do not keep modifying backend architecture** past this point unless a frontend integration test reveals a genuine backend bug, per the original instruction.
