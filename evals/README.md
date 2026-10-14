# Evaluation data

Only synthetic data is kept here: `reference.npz` holds vectors of fixed sample sentences and generated images,
and `reference_images/` holds the generated pictures.

The query sets and reports were built from a private archive (chats, call transcripts, work documents), so they
are not published. To rebuild them on your own index:

```bash
python tools/build_eval.py --out evals/queries.jsonl      # from your archive database
python tools/score.py                                      # writes evals/reports/<date>.md and .json
```
