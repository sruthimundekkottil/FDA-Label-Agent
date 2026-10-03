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

**Labeling procedure (to be updated honestly once review happens).**

Candidate questions are drafted by the coding agent from text that is actually
present in `data/processed/sections.jsonl`, and every drafted question is written
with `"verified": false`. A human must review each question, correct the
`gold_section` / `gold_answer_keypoints`, and flip `verified` to `true` before the
questions are used for any reported metric. Until that happens, the README must
describe the set as *unverified / model-drafted*.
