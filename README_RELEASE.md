# archive-index: bilingual author-review release

English: paper/manuscript/article-en.pdf (36 pages)
Ukrainian: paper/manuscript/article-uk.pdf (38 pages)

Code: Apache-2.0. Article, original figures and experimental records: CC BY 4.0.
LICENSING.md defines scope and third-party exceptions.

This compact export contains current code/tests, saved research JSON,
manuscript sources, tables/figures and PDFs. It contains no Git history,
raw corpus, private field inputs, indexes/caches or model weights.

With the recorded dependencies, Tectonic and the CMU fonts installed:

  PYTHONDONTWRITEBYTECODE=1 bash paper/run_all.sh quick --out paper/data/reproduction/review

This renders saved results; it does not repeat experiments or need model calls.
Full frozen-input statistical replay requires separately held query/run/grade,
corpus/vector inputs, which are not redistributed in this release while
third-party scope review remains pending. Scripts are included, and the
manuscript availability section discloses this boundary.

The text-relevance labels are model-generated. No human relevance audit,
Zenodo DOI, final author scientific approval or peer review is claimed.
