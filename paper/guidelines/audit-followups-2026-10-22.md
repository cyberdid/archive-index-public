# Audit follow-ups, 22 October 2026

These exploratory additions extend the completed no-Claude primary protocol in
`judging-protocol-2026-10-22.md`. They do not replace its frozen runs, primary qrels,
query population or corrected comparison family. Publication is still a separate
author decision.

## Encoder and coverage controls

`encoder_control.json` compares EG2 restricted to the four text collections with
E5. Both use the same 4,137 document candidates. The signed difference in the
control table is **system minus E5**. Controls did not form the original pool:
report unjudged@10, ordinary scores and condensed-list scores. Paired control
p-values are exploratory and unadjusted.

`covers_fix.json` preserves the frozen adaptive runs and compares the whole-word
coverage correction beside them. Do not regenerate the canonical primary report
with these new runs.

## Independent adjudication — deferred by the author

On 22 October 2026 the author explicitly chose to continue without Claude.
The partial blind pass (500 of 1,091 initial Gemini/Gemma 12B disagreements)
is archived for provenance only and excluded from every reported analysis.
It is not a dependency for completing the current study. Do not resume it when
quota resets; any future use requires a new explicit author decision.

The completed primary workflow remains reviewed Gemini with original Gemini and
Gemma 12B sensitivity analyses. The separate extension uses its documented
Gemini/Gemma protocol, with GPT comparison-only. No median-of-three variant is
computed from the partial Claude sheet.

## Human audit

`paper/data/judgments/human-audit.tsv` contains 150 shuffled, blinded pairs: 50
sampled from each primary grade. Only the author supplies `grade` (0/1/2) and
optional `note`. An AI agent must not fill these grades. Report agreement on this
stratified sample as such; unweighted sample agreement is not whole-pool agreement.

## Additional target types

The extension has 50 separate queries, IDs q201–q250, targeting 20 documents,
15 tables and 15 files from the code-repository part. Repository targets include
source files, README/documentation and a development log, not source code alone.
The generated 50 records contain 49 distinct wordings; one wording targets two files. Qwen 3.8 27B through local Ollama writes them from
indexed leads. It is neither the original query writer nor a relevance judge.
Require 3–10 words and at most five consecutive shared tokens. Store accepted and
rejected responses and a hash of each prompt; resume accepted items without
rewriting them. Never silently release fewer than 50 queries.

Run the same ten systems with the corrected coverage rule into a separate run
directory. Pool their top ten and prepare separate Gemini, Gemma 12B and GPT sheets.
Initial Gemini labels are primary for this extension, initial Gemma 12B labels are
the sensitivity analysis, and GPT remains comparison-only. Keep these labels and
metrics separate from the original 200 queries and their reviewed primary labels.

This extension examines target coverage and judge behaviour on a different query
population. It cannot identify an effect of query authorship: target types,
topics, languages and difficulty also differ. Do not claim it separates general
leniency from self-preference.

```bash
python -m paper.queries.extend
python -m paper.runs.run_all --queries paper/data/queries-ext.tsv --out paper/data/runs-extension
python -m paper.eval.extension prepare
python -m paper.judge.sheets judge --judge gemini --sheet paper/data/judgments-extension/gemini.tsv
python -m paper.judge.sheets judge --judge gemma12 --sheet paper/data/judgments-extension/gemma12.tsv
python -m paper.judge.sheets judge --judge gpt --sheet paper/data/judgments-extension/gpt.tsv
python -m paper.eval.extension evaluate
```

Use the research virtual environment and `PYTHONPATH=src:.` as in `paper/README.md`.
Do not repeat `prepare` on graded sheets.

## Costs and field study

Run latency/indexing/GGUF/ANN measurements sequentially. Do not start duplicate
model jobs alongside an existing timing run. Record desktop load and failures;
an error must not be reported as a completed stage. GPU and CPU indexing use
475-file and 95-file samples respectively; describe their composition and compare
throughput per chunk without presenting the wall-time ratio as a paired speedup.

GPU memory differences are device-wide changes from each stage's baseline; models
remain loaded between query systems, so these are not isolated model footprints.
ANN collections above the real vector population are synthetic. Save every
completed size with input hashes and an explicit completion flag; incomplete
scaling files must not be presented as the full planned experiment.

The private field runner exports counts, rates and confidence intervals only.
Exact-item ranks are chunk ranks, described-item success checks the expected
project, and absent-item success checks lexical abstention. These are operational
checks on author-written development queries, not independent relevance labels
or a user study. Keep all private query text, paths and snippets out of artifacts.
