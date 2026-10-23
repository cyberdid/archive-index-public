# UA-PAR relevance assessment: decision of 22 October 2026

This protocol supersedes the original Claude-primary plan. The author approved continuing without a Claude judge
on 22 October 2026. It was recorded after inspecting judge agreement and target/no-answer diagnostics, before
computing retrieval system rankings. It is a post hoc choice, not a preregistered protocol.

## Initial assessments and preserved evidence

- Primary: Gemini 3.8 Flash, CLI identifier `gemini-3.8-flash-medium`, through `agy`, batches of 50.
- Sensitivity assessment: Gemma 4 12B, Ollama `gemma4:12b`, digest prefix `4eb23ef187e2`, temperature 0,
  thinking disabled, context 2048.
- Ancillary open-weights assessment: Gemma 4 E4B, Q4_0, through llama.cpp.
- GPT-6-Luna xhigh created the 200 queries. Its grades are comparison evidence only and never enter the
  primary, sensitivity or reviewed qrels.
- Initial assessments were produced separately on independently shuffled sheets, with no other judge's grades
  or retrieval system identity shown. Gemini and Gemma are both Google models: separate assessments do not imply
  independence of training data or errors.

The four initial sheets contain 5572 grades each. The depth-10 pool is the union of ten retrieval systems.
The query and snippet for a pair must agree across sheets. Snippets show the document title and beginning, up
to 1200 characters; audio snippets include the source transcript. These are assessments of that evidence, not
of unseen full documents. The unchanged executable rubric is `GUIDELINE` in `paper/judge/llm_judge.py`.

Initial sheets, query definitions, corpus manifest, runs and available batch-call logs were copied to
`paper/data/judgments/snapshot-2026-10-22/` before retrieval evaluation. SHA-256 checksums and the snapshot time
are recorded in `paper/results/judging_inputs_initial.json`. Raw source material remains local under ignored
`paper/data/`; results must never require private field-study content.

## Disagreement review

Review every pair on which initial Gemini and Gemma 12B disagree. Preserve the original query and snippet,
and use the same relevance definitions (0 unrelated, 1 partly related, 2 direct answer). A separate Gemini
pass sees both initial grades under neutral A/B labels, with their order varied deterministically. It must
reassess the evidence; it is not instructed to prefer either grade or to improve agreement. Store the review
in a new TSV and log every cloud call, including response, model, conversation identifier and usage when supplied.

This is a within-model review by Gemini, not an independent third judge or human adjudication. Preserve all
initial grades. In reviewed qrels, retain agreed initial grades and replace only disagreement grades with
the complete review result. Refuse partial, invalid, duplicated or mismatched review sheets.

## Evaluation and sensitivity

Produce three separately labelled qrels/results variants: initial Gemini, initial Gemma 12B and reviewed Gemini.
The primary reported retrieval results use reviewed Gemini, accompanied by the two initial variants. Report
system rankings and whether pairwise ordering or the direction of differences changes across variants;
agreement between judges alone does not establish correctness. Do not select a qrels variant because it ranks
archive-index more favourably.

Use the same frozen retrieval runs and query population for every variant. The 30 deliberately unanswerable
queries are excluded from retrieval relevance means and assessed separately. Count answerable queries with no
positive pooled judgement explicitly; they receive zero under the existing metrics. Report per-category and
per-language scores, paired bootstrap intervals and paired sign-flip comparisons against `adaptive`, with
10000 resamples, seed 2026, and Holm correction over system comparisons within each metric and variant.

All systems' top-10 documents must be judged. Depth-10 pooling does not exhaust the corpus, and relevance labels
for ranks 11–100 are incomplete. `R@100` is recall relative to known relevant documents in this pool; it must not
be described as full-corpus recall. Report coverage of ranks 1–10 and 1–100 with the results.

## Corpus and query coverage

The frozen manifest records 7070 collected files: 2945 notes, 400 rendered documents, 150 tables, 1775 files from
code repositories, 1000 benchmark images and 800 speech clips. Not every collected repository file is indexable.
The full EG2 indexes contain 6054 documents (759 from the code part); the text-only E5 index contains 4137
(642 from the code part, with the benchmark images and speech omitted). These are different candidate universes,
not an identical-corpus encoder comparison. Counts are recorded in `paper/results/corpus_summary.json`.

All 170 answerable query targets are notes; the corpus is heterogeneous but target construction is note-centred.
The pool contains 134 of those targets. Target-grade diagnostics describe these 134 pooled targets, not the full
170-target set. Source-dataset image and speech retrieval experiments are separate from this text pool.

## Limits and author involvement

The author-review edition requested on 23 October 2026 intentionally uses model-generated text-relevance
labels without a human relevance audit. This is a scope decision for this edition, not a statement of human
verification. The prepared 150-pair, grade-balanced blinded sheet remains ungraded and available for optional
future work; it must not be described as a random 100-pair audit or as completed review. Source-dataset
multimodal pairings and implementation tests do not supply human labels for the text pool.

Target-document and designated no-answer diagnostics are construction-based plausibility checks, not a
complete gold standard. Preserve original labels, the within-model review and judge sensitivity. The original
fusion hypothesis is not an experimental finding: report stronger baselines and unfavourable ablations as
measured. Final author source/declaration review and separate publication approval remain required.
