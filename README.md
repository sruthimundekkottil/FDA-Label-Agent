# Evidence-Grounded RAG over FDA Drug Labels

Retrieval-augmented question answering over openFDA drug labels, with citations,
plus a small controlled experiment comparing chunking and retrieval strategies
on semi-structured medical documents.

> **For information only. Not medical advice.**

## Status

Under construction, milestone by milestone:

1. Scaffold — done
2. Data fetch + parse — in progress

All results tables in this README are filled in only after the corresponding
script has been run; anything not run is reported as "not run".

## How to run (so far)

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env           # then fill in LLM_* if you want real answers
python -m src.fetch_openfda
python -m src.parse_labels
```

## License + disclaimer

MIT (see `LICENSE`). This project uses public openFDA data and is **not** a
medical device, does not provide medical advice, and must not be used to make
clinical decisions.
