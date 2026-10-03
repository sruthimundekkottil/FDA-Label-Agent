# Evaluation set

`questions.jsonl` holds one JSON object per line:

```json
{
  "id": "q001",
  "question": "...",
  "drug": "...",
  "gold_section": "...",
  "gold_answer_keypoints": ["..."],
  "question_type": "factual | negation | table | abbreviation | cross-drug | multi-section | unanswerable",
  "verified": false
}
```

## How this set was built (honest procedure)

`questions.jsonl` was **drafted by a script**, not by a human:

```bash
python -m src.draft_eval_questions      # seed 42, deterministic
```

- Every question is generated from text that is really present in
  `data/processed/sections.jsonl`; `gold_answer_keypoints` are verbatim snippets
  from the matching section, truncated at 170 characters (nothing is invented).
- Table questions are generated from the parsed `adverse_reactions_table`
  records: the row label comes from the table, and the keypoint is the full
  parsed row with its values.
- Abbreviation questions are attached to the drug and section that mention the
  abbreviation most often (e.g. INR -> warfarin), using
  `ABBREVIATION_SECTION_PRIORITY` to break ties towards the more central section.
- Unanswerable questions have an empty `gold_section`; they are excluded from
  retrieval metrics and only test abstention behaviour.
- Cross-drug questions name two drugs but keep a single `drug` field: the first
  drug is the primary one used for retrieval scoring.

**Status: 0 of 40 questions are human-verified.** Every record carries
`"verified": false`.

| question_type | count | target |
| --- | ---: | ---: |
| factual | 15 | 15 |
| negation | 6 | 6 |
| table | 6 | 6 |
| abbreviation | 5 | 5 |
| unanswerable | 4 | 4 |
| multi-section | 2 | 2 |
| cross-drug | 2 | 2 |
| **total** | **40** | **40** |

## What a reviewer must do

For each line: check the question reads naturally, check that `gold_section`
really contains the answer, correct or trim `gold_answer_keypoints` (especially
for table rows, which contain raw column text), then set `"verified": true`.

The evaluation scripts report how many questions are verified vs unverified and
accept a `--verified-only` flag; until questions are verified, any reported
number must be described as computed on **model-drafted, unverified** questions.
