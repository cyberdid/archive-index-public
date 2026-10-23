# Scientific claim review

Machine-assisted review on 22–23 October 2026. Human approval remains pending.
The original frozen retrieval scores and original qrels are preserved.

| Claim / location | Evidence | Disposition |
| --- | --- | --- |
| Full index contains 96,908 embedding vectors; method and ANN description | Read-only `chunks` SQL: 96,908 rows, 86,726 with dimension > 0 | **Revise:** 96,908 raw chunks; ANN begins with 86,726 real vectors |
| GGUF embeds 94,970 text chunks | `text_chunk_rows` filters dimension > 0 and text collections; SQL count is 84,788, including 21 image-embedded PDF-page rows | **Revise:** 84,788 collection rows; 84,767 text-input rows. GGUF encodes stored text for the 21 page rows, so input modality also changes |
| Query latency includes document collapse/path resolution | `paper/runs/systems.py`: elapsed time appended before `to_document_run` | **Revise:** embedding and chunk search only; process counters cover a broader pass |
| Q4 reduces intrinsic retrieval quality | Ordinary Δ −0.0182, but 9.88% unjudged@10; condensed Δ +0.0042 with CI spanning zero | **Weaken:** report both evaluations. The frozen pool cannot identify intrinsic loss; backend/precision also change |
| No significant f16/Q8 difference proves equivalence | Paired CIs span zero; no equivalence design or margin | **Remove implication:** small point differences with unresolved uncertainty |
| Candidate restriction costs 0.002 nDCG | Exact difference −0.00147847; rounded table means .633 and .631 | **Revise:** use exact contrast, rounded to −0.0015, not subtraction of display values |
| Encoder control proves E5 always better | Reviewed Gemini contrast differs from Gemma12; exploratory tests; incomplete pool | **Weaken:** label-specific exploratory evidence on this archive only |
| Inflection means surface-form search misses most forms | No quantified coverage evidence | **Weaken:** it can miss documents using other forms |
| No existing collection models the same setting | No systematic literature review establishing absence | **Revise:** describe UA-PAR's constructed setting without a universal novelty claim |
| HNSW/Faiss make dense search sublinear | No worst-case complexity proof for this setup | **Revise:** they provide approximate vector search; report actual measured cost |
| The optimized NumPy scaling reference is the archive-index implementation | Reference uses `argpartition`; source `semantic_scored` gathers candidate rows, fully sorts and constructs 400 tuples | **Revise and measure:** keep the optimized reference and add a sequential audit calling the actual source method; exclude only cache-version polling |
| Every strict-ID disagreement between exact methods is an approximation error | Equal scores and floating arithmetic can change the IDs selected at a boundary | **Revise:** report strict-ID agreement and a separate reference-score threshold diagnostic; HNSW remains strict-ID recall |
| Text is never transmitted over a network | Default local model; optional `ARCHIVE_LLAMA_URL` can name an external server | **Revise:** local processing depends on backend configuration |
| Privacy filters guarantee removal of secrets | Known patterns; images/audio not content-redacted | **Weaken:** patterns are a limited safeguard, not a guarantee |
| Private field checks establish general user effectiveness | One author's development archive and queries; expected phrase/project checks | **Keep only scoped claim:** operational behaviour in this deployment; no human relevance/user-study claim |
| All-success bootstrap intervals establish perfect population performance | Constant empirical samples give degenerate percentile intervals | **Remove implication:** explicitly disclose the degeneracy and sampling limitation |
| All 15 extension repository targets are source code | 11 source files, 3 Markdown files, 1 text log | **Revise:** repository files; retain exact suffix breakdown |
| All 50 extension records are independent wordings | q239 and q249 have the same normalized wording | **Revise:** retain all frozen records and add a deterministic 49-wording sensitivity; disclose the dependence in 50-record intervals |
| Independent Claude adjudication is required for completion | Author explicitly deferred Claude | **Remove dependency:** preserve partial sheet only as excluded provenance |
| Citation metadata alone proves the cited proposition | Metadata may disagree with publisher PDFs and says little about support | **Keep gate open:** inspect original content and record locators; never set human verification true |

## Bibliography metadata conflicts

- Sentence-BERT: the [ACL publisher PDF](https://aclanthology.org/D19-1410.pdf),
  first-page proceedings header, gives pages **3982–3992**. The DOI metadata
  returned 3980–3990. The publisher PDF controls the corrected BibTeX entry.
- BM25: DOI metadata returned volume 4(1–2), pages 1–174. The original article's
  title page, visible in the indexed [author-hosted paper](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf),
  gives **3(4), 333–389 (2009)**. The author-hosted original PDF was retrieved on 23 October 2026; its title
  pages, abstract, introduction and BM25 formulation confirm this correction.
  Its SHA-256 and locators are recorded in `source_manifest.json`. The publisher
  endpoint remains HTTP 403. Final author source approval is still separate.
- Weighted kappa: the DOI registry drops the word “with” from the title. The
  publicly readable original-article transcription (title page, abstract and
  pp. 213–216), corroborated by the NLM bibliographic record, includes it.
  The BibTeX title now follows that original wording. Weighted disagreement is
  supported by the original definition; linear weights are our stated choice.
- Stuff I've Seen: the DOI registry truncates the title. The author institution's
  publication page supports the full subtitle retained in the bibliography.

All 27 entries now have scoped content-support records. Most background claims
are checked against original abstracts or author-institution publication
abstracts, not an implied reading of every full paper. BM25's original PDF has now been obtained and the relevant title pages,
abstract, introduction and formulation inspected. Three metadata corrections and the shortened-title case
are retained in `source_manifest.json`. No human verification is asserted.

ANN and extension results are complete. The current bilingual interpretation
was reviewed against the frozen JSON and timer/ranking code. Source-dataset
multimodal pairings are now consistently distinguished from independent
human relevance judgements. Final author reading remains; this audit is not a
submission certificate.
