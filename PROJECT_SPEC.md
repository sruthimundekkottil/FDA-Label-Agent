# Project Spec: Evidence-Grounded RAG over FDA Drug Labels (with a Retrieval Experiment)

> **Instructions to the coding agent (Codex):** Build this project end to end in a fresh git repo, following this spec. Work in milestones, make one git commit per milestone with a meaningful message, and keep every script runnable from the command line. Do not fabricate any results, metrics, or numbers. All numbers in the README must come from actually running the evaluation scripts. If something cannot be completed, say so in the README under "Limitations / Not done". Prefer simple, readable, well-commented code over clever code.

---

## 1. Goal

Build a Retrieval-Augmented Generation (RAG) assistant that answers questions about FDA drug labels with **source citations**, and treat it as a **small controlled experiment**:

> **Research question:** Which chunking and retrieval strategies work best on semi-structured medical documents (drug labels)?

The deliverables are (a) a working Streamlit app, (b) an evaluation harness with a results table, (c) a failure analysis, and (d) a README written like a mini research report.

The thing that makes this project stand out is **not** the chatbot. It is the **evaluation and failure analysis**. Prioritise those.

## 2. Scope (strictly)

**In scope (core, must finish):**
- One data source: **openFDA Drug Label API**.
- About 25 to 30 common drugs.
- 2 chunking strategies (fixed-size, section-aware) plus 1 optional (table-aware).
- 3 retrieval strategies (vector-only, BM25-only, hybrid via Reciprocal Rank Fusion) plus an optional cross-encoder reranker.
- A hand-labeled evaluation set (target 40, minimum 30 questions).
- Metrics: Recall@k, MRR, citation accuracy, and (optional) LLM-judged faithfulness.
- Streamlit UI.
- Deployment to Hugging Face Spaces or Streamlit Community Cloud.
- README with architecture diagram, results table, failure analysis.

**Out of scope (do NOT build):** agents, MCP server, ClinicalTrials.gov, PubMed, fine-tuning, multi-cloud benchmarking, Azure deployment, authentication, databases beyond local files. Mention them only as "Future work" in the README.

## 3. Tech stack

- Python 3.10+
- `requests` (openFDA fetch)
- `sentence-transformers` for embeddings. Default model: `BAAI/bge-small-en-v1.5` (fallback: `sentence-transformers/all-MiniLM-L6-v2`)
- `faiss-cpu` for the vector index (Chroma is an acceptable alternative)
- `rank_bm25` for lexical retrieval
- Optional reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers
- `streamlit` for the UI
- `pandas`, `numpy`, `matplotlib` for results tables and plots
- `python-dotenv` for config
- `pytest` for a few basic tests
- LLM for answer generation: **provider-agnostic**. Implement a thin wrapper `llm.py` that talks to any OpenAI-compatible chat completions endpoint, configured through environment variables (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`). This lets the user plug in OpenAI, Azure OpenAI, Groq, a local Ollama server, etc., without code changes. Include a `--no-llm` / mock mode so retrieval evaluation works with no API key at all.

## 4. Repository structure

```
drug-label-rag/
├── README.md
├── PROJECT_SPEC.md            # this file
├── requirements.txt
├── .env.example               # LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, EMBED_MODEL
├── .gitignore                 # data/raw, data/index, .env, __pycache__
├── configs/
│   └── drugs.txt              # one generic drug name per line
├── data/
│   ├── raw/                   # raw openFDA JSON per drug (gitignored if large)
│   ├── processed/             # cleaned sections as JSONL
│   └── eval/
│       ├── questions.jsonl    # hand-labeled eval set
│       └── README.md          # how the set was built and labeled
├── src/
│   ├── fetch_openfda.py       # download labels
│   ├── parse_labels.py        # extract + clean sections
│   ├── chunking.py            # fixed-size, section-aware, table-aware
│   ├── indexing.py            # build FAISS + BM25 indexes per chunking strategy
│   ├── retrieval.py           # vector, bm25, hybrid (RRF), optional rerank
│   ├── llm.py                 # OpenAI-compatible client + mock mode
│   ├── rag.py                 # prompt building + answer + citations
│   ├── evaluate_retrieval.py  # Recall@k, MRR across configurations
│   ├── evaluate_answers.py    # citation accuracy + optional faithfulness
│   ├── failure_analysis.py    # dump failed queries for manual inspection
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── results/
│   ├── retrieval_results.csv
│   ├── answer_results.csv
│   ├── figures/               # bar charts
│   └── failures.md            # curated failure analysis
├── tests/
│   ├── test_chunking.py
│   └── test_retrieval.py
└── docs/
    └── architecture.md        # Mermaid diagram
```

## 5. Milestones (one commit each)

1. **Scaffold:** repo structure, requirements, .gitignore, .env.example, empty modules.
2. **Data fetch + parse:** download labels, extract sections, save processed JSONL.
3. **Chunking:** implement strategies with tests.
4. **Indexing + retrieval:** build indexes, implement vector/BM25/hybrid (+ optional rerank).
5. **RAG pipeline:** prompt, generation, citations, CLI demo.
6. **Eval set + retrieval evaluation:** questions file, metrics script, results CSV.
7. **Answer evaluation + failure analysis:** citation accuracy, optional faithfulness, curated failures.
8. **Streamlit UI.**
9. **README, architecture diagram, plots, deployment notes.**

## 6. Data: openFDA drug labels

**Endpoint:** `https://api.fda.gov/drug/label.json`

Example query:
```
https://api.fda.gov/drug/label.json?search=openfda.generic_name:"metformin"&limit=1
```

Notes:
- No API key needed for light use (rate limited). Support an optional `OPENFDA_API_KEY` env var. Add a short `sleep` between requests and basic retry logic.
- Each result is one label. Most fields are **arrays of strings**; join them.
- Prefer labels where `openfda.generic_name` matches and the label has the key sections. If several labels exist for a drug, pick the first with the most populated sections, and record the `set_id` / `id` and `effective_time`.

**Drug list (`configs/drugs.txt`)**, about 25 to 30 common drugs across classes, for example:
metformin, lisinopril, atorvastatin, amlodipine, metoprolol, losartan, simvastatin, warfarin, apixaban, clopidogrel, omeprazole, levothyroxine, sertraline, escitalopram, gabapentin, ibuprofen, acetaminophen, amoxicillin, azithromycin, ciprofloxacin, prednisone, insulin glargine, empagliflozin, semaglutide, albuterol, montelukast, tramadol, alprazolam, furosemide, hydrochlorothiazide.

(Drop any that fail to fetch cleanly. Include a few drugs with boxed warnings, e.g. warfarin, ciprofloxacin, sertraline, tramadol, since boxed warnings make good test questions.)

**Sections to extract** (openFDA field names), when present:
- `boxed_warning`
- `indications_and_usage`
- `dosage_and_administration`
- `dosage_forms_and_strengths`
- `contraindications`
- `warnings_and_cautions` (or `warnings` for older-format labels)
- `adverse_reactions`
- `drug_interactions`
- `use_in_specific_populations` (and `pregnancy`, `pediatric_use`, `geriatric_use` if separate)
- `overdosage`
- `description`
- `clinical_pharmacology`
- Table fields such as `adverse_reactions_table` or other `*_table` fields when present (these hold table content and are useful for the table-aware strategy).

**Cleaning (`parse_labels.py`):**
- Join array strings, normalise whitespace, strip repeated headings/numbering artefacts where easy.
- Keep a record per section: `{drug, set_id, section, text, source_field, effective_time}`.
- Save to `data/processed/sections.jsonl`.
- Print a summary table: number of drugs, sections per drug, total characters.

## 7. Chunking strategies (`chunking.py`)

Every chunk must carry metadata: `chunk_id`, `drug`, `set_id`, `section`, `strategy`, `text`, `char_start`, `char_end`. Prepend a short header to the text used for embedding, e.g. `"Drug: metformin | Section: contraindications\n..."` (make this toggleable so it can be tested as an ablation).

1. **`fixed`**: sliding window by characters or tokens (default 800 characters, 150 overlap), applied to each drug's concatenated label text, ignoring section boundaries. Section metadata is assigned by whichever section the chunk's start falls in (or `mixed`).
2. **`section`**: one chunk per section; if a section exceeds a max length (default about 1500 characters), split it on paragraph/sentence boundaries with overlap, keeping the section label on all pieces.
3. **`table_aware`** (optional but desirable): same as `section`, but table content (from `*_table` fields or detected table-like blocks) is kept as its own chunk, never split mid-row, and linked to its parent section via metadata. Convert table text to a readable row-wise format (e.g. `"Adverse reaction: nausea | Drug: 25% | Placebo: 9%"`) if the structure can be recovered; otherwise keep it as plain text.

Write unit tests that check: no empty chunks, overlap works, section metadata is preserved, table chunks are not split mid-row.

## 8. Retrieval configurations (`retrieval.py`)

Build a separate index per chunking strategy. Implement:

- **`vector`**: FAISS cosine similarity (normalised embeddings, inner product). For bge models, prefix queries with the recommended query instruction.
- **`bm25`**: `rank_bm25` over simple tokenised text (lowercase, alphanumeric tokens).
- **`hybrid`**: Reciprocal Rank Fusion of vector and BM25 rankings, `score = sum(1 / (k + rank))` with `k = 60`. Retrieve top 20 from each, fuse, return top-k.
- **`hybrid_rerank`** (optional): take the top 20 from hybrid, rerank with the cross-encoder, return top-k.

All retrievers share one interface: `retrieve(query, k, drug_filter=None) -> list[Chunk with score]`.

Also implement an optional **drug filter** (restrict retrieval to a named drug) and an **auto-detect drug from query** helper (simple string match against the drug list). Evaluate with and without the filter, since cross-drug confusion is an interesting failure mode.

**Experiment grid:** chunking strategies x retrieval strategies, evaluated at k in {1, 3, 5, 10}.

## 9. RAG generation (`rag.py`, `llm.py`)

- Retrieve top-k (default 5) chunks.
- Build a prompt that: (a) instructs the model to answer **only** from the provided context, (b) requires citations in the form `[drug | section | chunk_id]` after each claim, (c) tells the model to say "The provided labels do not contain this information" when the context is insufficient, (d) forbids medical advice beyond what the label states.
- Return a structured object: `{answer, citations: [...], retrieved_chunks: [...], latency_ms, prompt_tokens, completion_tokens}` (token counts if the API returns them).
- Try at least **two prompt variants** (e.g. plain instruction vs. structured JSON output with explicit citation fields) and compare them on the eval set. Keep this small.
- Include a clear disclaimer in the UI: *"For information only. Not medical advice."*
- Mock mode: return the top chunk text as the "answer" so the pipeline and UI can be tested without an API key.

## 10. Evaluation set (`data/eval/questions.jsonl`)

**Important:** the user will review and correct the labels by hand. Codex may *draft* candidate questions from the processed sections to speed things up, but must mark them `"verified": false`, and the README must state the true labeling procedure (which questions were drafted by a model, which were verified by a human). Never claim a human verified labels unless the user confirms.

**Target:** 40 questions (minimum 30), spread across drugs and sections.

**Schema (one JSON object per line):**
```json
{
  "id": "q001",
  "question": "What are the contraindications of warfarin?",
  "drug": "warfarin",
  "gold_section": "contraindications",
  "gold_answer_keypoints": ["pregnancy (except mechanical heart valves)", "hemorrhagic tendencies"],
  "question_type": "factual | negation | table | abbreviation | cross-drug | multi-section | unanswerable",
  "verified": false
}
```

**Question-type mix (aim for):**
- about 15 plain factual (indications, dosage, contraindications, boxed warnings)
- about 6 negation / exclusion style (e.g. "Which patients should NOT take X?", "What conditions is X not recommended for?")
- about 6 table-dependent (e.g. adverse reaction incidence rates, dose adjustment tables)
- about 5 abbreviation-heavy (e.g. "CrCl", "MAOI", "QT prolongation", "NSAID")
- about 4 unanswerable (information not in the label; the correct behaviour is to say so)
- about 4 multi-section or cross-drug (e.g. drug interactions between two named drugs)

**Retrieval ground truth:** a retrieved chunk counts as a **hit** if its `drug` matches `drug` **and** its `section` matches `gold_section` (for `fixed` chunking, use the section metadata assigned to the chunk and also count overlap with the gold section's character range). Unanswerable questions are excluded from retrieval metrics and used only for answer behaviour.

## 11. Metrics

**Retrieval (`evaluate_retrieval.py`):**
- **Recall@k** (k = 1, 3, 5, 10): fraction of questions with at least one hit in the top-k.
- **MRR**: mean reciprocal rank of the first hit.
- Report overall and **broken down by question_type**.
- Save `results/retrieval_results.csv` with columns: `chunking, retrieval, k, drug_filter, recall, mrr, n_questions, question_type`.
- Generate a bar chart (`results/figures/`) of Recall@5 per configuration.

**Answer quality (`evaluate_answers.py`)** (run on the best retrieval configuration plus a baseline to save API cost):
- **Citation accuracy:** fraction of answers whose cited `[drug | section]` pairs match the gold drug and section, and whose cited chunk IDs actually exist in the retrieved set (detects fabricated citations).
- **Abstention accuracy:** on unanswerable questions, does the answer decline?
- **Faithfulness (optional):** LLM-as-judge that checks each claim in the answer against the retrieved context and outputs supported / unsupported counts. Report the judge model used and note that LLM-judged scores are approximate.
- **Keypoint coverage (optional):** fraction of `gold_answer_keypoints` mentioned in the answer (simple judge or string matching).
- Track average latency and token usage per query.
- Save `results/answer_results.csv`.

Set a fixed random seed and fixed temperature (0) so results are reproducible. Record the embedding model name, LLM name, and library versions in the results output.

## 12. Failure analysis (`failure_analysis.py` and `results/failures.md`)

- Script dumps every question where the best configuration missed (no hit at k=5) or produced a bad citation, along with the retrieved chunks and gold section.
- Then write **3 to 5 curated failure cases** in `results/failures.md`, each with: the question, what was retrieved, what should have been retrieved, and a **hypothesis for why** it failed. Likely categories to look for:
  - table content split or flattened badly
  - negation ("not recommended", "excluded") retrieved as if affirmative
  - abbreviation mismatch between query and label text
  - cross-drug confusion (similar drugs, same class)
  - sections that overlap in content (warnings vs. contraindications)
- Codex should generate the raw dump and draft the analysis, but flag that the user must read and confirm the conclusions.

## 13. Streamlit app (`app/streamlit_app.py`)

- Text box for the question; optional dropdown to restrict to a drug.
- Sidebar controls: chunking strategy, retrieval strategy, top-k.
- Display: the answer with citations, then an expandable list of retrieved chunks (drug, section, score, text).
- Show latency and token usage.
- Show the "not medical advice" disclaimer.
- Load pre-built indexes from disk (do not rebuild on every run). Cache with `st.cache_resource`.
- Handle missing API key gracefully by falling back to mock mode with a visible notice.
- For deployment, commit the small processed data and a prebuilt index if size allows (otherwise build on first startup and cache).

## 14. Deployment

- Target: **Hugging Face Spaces (Streamlit)** or **Streamlit Community Cloud**.
- Provide `requirements.txt` pinned to working versions and a short "Deploying" section in the README.
- API keys must come from the platform's secrets, never from the repo.
- Add a `Dockerfile` as a bonus (optional), so the app can be run with one command.

## 15. README requirements (write it like a mini research report)

Sections, in this order:
1. **Title + one-line summary + demo link/screenshot**
2. **Motivation:** why medical labels are a hard test for RAG (tables, negations, abbreviations, long documents)
3. **Research question and hypotheses**
4. **Data:** source, drugs used, number of labels/sections/chunks, date fetched
5. **Method:** chunking strategies, retrieval strategies, prompts, models (with an architecture diagram in Mermaid)
6. **Evaluation set:** how it was built, question types, labeling procedure (be honest about what was model-drafted vs. human-verified)
7. **Results:** tables and figures generated from `results/*.csv` only. Include the best configuration and the improvement over the baseline
8. **Failure analysis:** 3 to 5 cases with hypotheses
9. **Limitations:** small eval set, one label version per drug, LLM-judge noise, no clinical validation, not medical advice
10. **Future work:** table-aware chunking improvements, ClinicalTrials.gov, agentic verification step, MCP tool, Azure deployment
11. **How to run:** install, set `.env`, fetch data, build indexes, run evaluation, launch app
12. **License + disclaimer**

Do not put any placeholder numbers in the README. If a metric was not computed, write "not run".

## 16. Quality bar / acceptance criteria

The project is done when:
- [ ] `python -m src.fetch_openfda && python -m src.parse_labels` produces `data/processed/sections.jsonl` for at least 20 drugs
- [ ] Indexes build for at least `fixed` and `section` chunking
- [ ] `vector`, `bm25`, and `hybrid` retrieval all run from one interface
- [ ] `python -m src.evaluate_retrieval` writes `results/retrieval_results.csv` and a figure
- [ ] Eval set has at least 30 questions with `gold_section` labels
- [ ] `python -m src.evaluate_answers` runs (or clearly documents that it needs an API key)
- [ ] `results/failures.md` contains at least 3 analysed failures
- [ ] Streamlit app runs locally with `streamlit run app/streamlit_app.py`, including in mock mode
- [ ] Tests in `tests/` pass with `pytest`
- [ ] README follows section 15, with all numbers traceable to the results files
- [ ] Git history has one commit per milestone with clear messages

## 17. Priority order if time runs short

1. Data fetch + section parsing
2. `fixed` vs `section` chunking
3. `vector` vs `hybrid` retrieval
4. Eval set of 30 questions + Recall@k table
5. Streamlit app (mock mode is fine)
6. README with honest results
7. Failure analysis
8. Citation accuracy / faithfulness
9. Table-aware chunking, reranker, prompt variants
10. Deployment

If something must be cut, cut from the bottom of this list, and document it under "Limitations / Not done" in the README.

## 18. Constraints and ethics

- Never present outputs as medical advice. Keep the disclaimer visible.
- Use only public openFDA data and respect its terms of use and rate limits.
- Do not hard-code API keys. Keep `.env` out of git.
- Do not invent or backfill metrics. Do not alter git commit timestamps.
