# Relevance guideline (UA-PAR)

Judges (decision of 22 October 2026): Gemini 3.8 Flash (primary), Gemma 4 12B (sensitivity assessment), and Gemma 4 E4B
(ancillary assessment), initially graded separately on independently shuffled copies of the same pool. GPT-6-Luna
wrote the queries; its grades are comparison only. A separate Gemini pass reviews Gemini/Gemma 12B disagreements
with both initial grades visible. This is a within-model review, not an independent third judge. The initial sheets
remain unchanged, and retrieval results are reported for both initial judges and reviewed Gemini. Gemini and Gemma
share a provider, so their agreement may include shared biases. No human sample audit has yet been recorded.

The full procedure, decision timing, provenance, metrics and limitations are in
[judging-protocol-2026-10-22.md](judging-protocol-2026-10-22.md).

Judge at **document level**: the snippet stands for the document; judge what the document is about, not single words.

| Grade | Meaning | Example (query → document) |
|---|---|---|
| 2 | answers the query directly, or is the very document the query describes | «як працює супергетеродинний приймач» → note explaining superheterodyne reception |
| 1 | the same topic, useful context, but not the answer | «як працює супергетеродинний приймач» → note on the history of radio receivers |
| 0 | unrelated, or shares only words | «як працює супергетеродинний приймач» → note on a football club called «Приймач» |

Rules:
1. Language does not matter: a Russian document can be 2 for a Ukrainian query (cross-lingual category).
2. For technical queries the identifier must refer to the same thing (3GPP TS 36.211 ≠ TS 36.213).
3. Images and recordings are judged from their caption or transcript as given in the sheet.
4. When unsure between two grades, choose the lower one.

The text given to the LLM judges is `GUIDELINE` in `paper/judge/llm_judge.py`; it is this table in short form.
