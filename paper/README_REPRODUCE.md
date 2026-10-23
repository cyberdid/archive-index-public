# Reproducing the archive-index research draft

This is a private research draft, not an approved data release. Publication
identifiers have not been assigned. Run commands from the repository root.
The original 200-query runs and reviewed-Gemini labels are frozen; corrected
code runs and the 50-record extension are separate experiments.

## Environment

Use Python 3.13 and the project with its `research` dependencies. The measured
environment is recorded in `paper/environment/ENVIRONMENT.txt`, with exact
observed distribution versions in `requirements-measured.txt`. Those pins are
an observation, not yet a hash-locked fresh-install certificate. ANN uses a
separate CPU environment, recorded in `requirements-ann.txt` and the ANN JSON.
The hash-locked solver outputs are `requirements-research-lock.txt` (109
packages, constrained to the observed main versions) and
`requirements-ann-lock.txt` (the separate ANN environment). They include wheel
or source-distribution SHA-256 hashes. A fresh full CUDA installation has not
been performed as part of this draft's packaging.
The PDF build also needs Tectonic, the CMU fonts used by `tex/preamble.tex`, and
Poppler for compliance inspection. The measured Tectonic version is 0.15.0;
its bibliography rerun warnings are discussed in `manuscript/REVIEW.md`.

```bash
export PYTHONPATH="$PWD/src:$PWD"
PAPER_PYTHON=${PAPER_PYTHON:-python3}
```

## Automatic isolated runs

```bash
PAPER_PYTHON="$PAPER_PYTHON" bash paper/run_all.sh quick
PAPER_PYTHON="$PAPER_PYTHON" bash paper/run_all.sh full

# Continue a specific existing run, verifying its inputs and completed outputs.
PAPER_PYTHON="$PAPER_PYTHON" bash paper/run_all.sh full \
  --out paper/data/reproduction/your-run-id --resume
```

Each run has an immutable `inputs/` snapshot, a separate writable `work/` tree,
per-attempt command logs, a JSON manifest with exit codes and SHA-256 hashes,
and `SHA256SUMS`. A process lock rejects concurrent use of one run directory.
Resume rejects changed source bytes, inputs, Python environment, completed
outputs or logs; it does not mix revisions silently.

`quick` regenerates tables/figures and both PDFs from archived result JSON.
`full` additionally validates the frozen input lock, reconstructs qrels,
recomputes all per-judge original-query statistics, agreement diagnostics,
candidate/coverage controls, GGUF quality and both extension populations, then
requires exact equality with the archived reports and canonical qrels. ANN
vector-file hashes are checked. Neither mode calls a judge or repeats physical
measurements. Imported resource/multimodal/ANN/private-field results are named
in the manifest; fresh public measurement is a separate recipe below.

Successful reproduction remains separate from release approval. The manifest
records actual PDF compliance findings and always leaves human/release approval
false. It does not turn an unapproved draft into a publication certificate.

## Rebuild tables and figures without executing models

```bash
"$PAPER_PYTHON" -m paper.report.generate
"$PAPER_PYTHON" -m paper.report.validate
PAPER_PYTHON="$PAPER_PYTHON" bash paper/manuscript/tex/build.sh all
"$PAPER_PYTHON" -m paper.release.compliance preprint
```

The standalone build validates measured-source hashes and generated content
before invoking Tectonic. It fails if source JSON changes without regeneration.
The compliance command additionally rejects visible pending markers. A draft
with pending release decisions can compile while failing that release gate;
compilation does not imply scientific or author approval.

## Recompute frozen-query statistics in a separate output directory

This requires the archived **public-benchmark** snapshot and graded Gemini
review under `paper/data/judgments/`. Raw data are ignored by Git and must be
supplied separately. No judging command or cloud call is part of this recipe.

```bash
paper_reproduction_dir=$(mktemp -d "$PWD/paper/data/reproduction.XXXXXXXX")
mkdir -p "$paper_reproduction_dir/paper/results"
cp paper/results/*.json "$paper_reproduction_dir/paper/results/"
"$PAPER_PYTHON" -m paper.judge.finalize validate \
  > "$paper_reproduction_dir/input-validation.json"
"$PAPER_PYTHON" -m paper.eval.judge_sensitivity \
  --review paper/data/judgments/gemini-review.tsv \
  --out "$paper_reproduction_dir/paper/results"
cp "$paper_reproduction_dir/paper/results/ua_par_gemini_reviewed.json" \
   "$paper_reproduction_dir/paper/results/ua_par.json"
"$PAPER_PYTHON" -m paper.report.generate \
  --results "$paper_reproduction_dir/paper/results" \
  --out "$paper_reproduction_dir/paper/manuscript/generated"
"$PAPER_PYTHON" -m paper.report.validate \
  --results "$paper_reproduction_dir/paper/results" \
  --out "$paper_reproduction_dir/paper/manuscript/generated"
```

Only frozen-query statistics are recomputed by this short recipe. Resource,
multimodal and private-field aggregates are explicitly imported measured
artifacts. Fresh measurement is a different operation and cannot be inferred
from rebuilding their tables. Keep command logs, exit codes, source revision
and input/output SHA-256 with every reproduction run.

## Fresh public measurements

Use an isolated checkout with its own writable indexes and result directory.
Do not point an indexing command at a canonical database or replace the frozen
retrieval runs. The four indexes are built by `paper.runs.run_all --build` from
the corpus created by `paper.corpus.collect` and `paper.corpus.build`; inspect
their `--help` before selecting a new corpus location. Corpus source URLs,
licenses and checksums are in `paper/corpus/` and the per-file manifest.

The experiment entry points are:

| Measurement | Entry point | Inputs / interpretation |
| --- | --- | --- |
| CPU/GPU query cost | `python -m paper.eval.run_efficiency queries --device cpu` or `gpu` | Existing public indexes and 200 queries; full warm-up; query embedding and chunk search only |
| CPU/GPU indexing | `python -m paper.eval.run_efficiency index --device cpu` or `gpu` | Different fixed sample sizes; throughput is not a paired hardware speedup |
| GGUF retrieval | `python -m paper.eval.run_efficiency gguf` | Existing public index, three GGUF builds and local llama-server; version and model hashes must match cache fingerprints |
| Multimodal tasks | `python -m paper.eval.multimodal` | Collected XM3600/FLEURS assets and paired source labels; transcript text mode is a ceiling condition |
| ANN vectors | `python -m paper.eval.ann_scaling export` | Exports real vectors and 200 embedded queries; run once before CPU scaling |
| Synthetic ANN scaling | `uv run --no-project --python 3.13 --with faiss-cpu==1.15.1 --with numpy==2.5.4 python -m paper.eval.ann_scaling` | Four synthetic sizes, NumPy top-k, Faiss flat and HNSW; atomic checkpoints |
| Actual archive vector path | `uv run --no-project --python 3.13 --with faiss-cpu==1.15.1 --with numpy==2.5.4 --with pymorphy3==2.0.6 python -m paper.eval.ann_reference_audit` | Run after the preceding benchmark; same vectors, actual candidate gather/full sort, separate exact-ID agreement audit |
| Separate extension analysis | `python -m paper.eval.extension evaluate` | Frozen 50-record extension, prepared hashes, complete initial Gemini/Gemma sheets; GPT is ancillary only |

The efficiency module's GGUF/server locations currently follow the measurement
machine's local installation; configure or adapt those locations in a new
isolated checkout and record the change. Model identifiers and pinned revisions
are part of the evidence, not a reason to substitute a different model silently.
Run measurement jobs serially and record desktop load. The observed laptop
session was not an isolated performance laboratory.

## Populations, privacy and release preparation

- `python -m paper.corpus.summary` reads public indexes without changing them.
  Raw chunks, vectors, text-collection vectors and actual text-input vectors
  are different populations. The 84,788 GGUF rows include 21 PDF-page rows whose
  full-model input is an image but whose GGUF input is stored text.
- The original query set has 170 answerable queries and 30 no-answer queries.
  The latter are excluded from retrieval means. The separate extension has
  50 records and 49 wordings; its deterministic unique-wording sensitivity
  keeps the lowest query ID, without selection by score.
- Private field inputs are not part of the public recipe. Only
  `field_aggregate.json` may be imported. Never copy private queries, paths,
  retrieved snippets or private indexes into a data bundle.
- Frozen corpus manifests contain local absolute paths. A public export must
  normalize those paths and record both original/exported hashes and the
  transformation. It must not falsely present normalized bytes as the original
  snapshot. Recomputed public provenance can differ while metrics remain equal.
- Before any publication, scan the actual prepared bundle and extracted PDF
  text with the private deny list; preserve source licenses and attribution.
  `COMPLETION.md`, `CLAIM_AUDIT.md` and `source_manifest.json` record remaining
  scientific, human-review and release gates.
