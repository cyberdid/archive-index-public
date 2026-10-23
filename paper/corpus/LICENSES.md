# UA-PAR sources and licences

Checked against the primary sources on 2026-10-19.

| Source | Part | Licence | Attribution | Checked at |
|---|---|---|---|---|
| Ukrainian Wikipedia | notes, documents | CC BY-SA 4.0 (and GFDL) | URL of every article, in the manifest | Wikimedia Terms of Use, section 7 |
| Russian Wikipedia | notes, documents | CC BY-SA 4.0 (and GFDL) | URL of every article, in the manifest | Wikimedia Terms of Use, section 7 |
| data.gov.ua datasets | tables | per dataset; only `cc-by`, `cc-zero`, `odc-by`, `odc-pddl`, `cc-by-sa` are taken | dataset URL and publisher, in the manifest | CKAN `license_id` of each dataset |
| GitHub repositories (topic `ukrainian`) | code | per repository; only MIT and Apache-2.0 are taken | repository URL; the repository's LICENSE file is kept | GitHub API `license.spdx_id` |
| Crossmodal-3600 captions | images | CC BY 4.0 (Google LLC) | Thapliyal et al., EMNLP 2022 | google.github.io/crossmodal-3600 and web-data/README.md |
| Crossmodal-3600 images (Open Images) | images | CC BY 2.0, all 3 600 images | author and URL per image, from `image_attributions.csv` | `image_attributions.csv`, column `License` |
| FLEURS `uk_ua`, `ru_ru` test split | speech | CC BY 4.0 | Conneau et al., 2022 | dataset card on huggingface.co/datasets/google/fleurs |

Documents (PDF, DOCX, PPTX) are rendered by `paper/corpus/build.py` from Wikipedia articles that are not used as notes,
so the archive has its usual document formats while every text stays CC BY-SA 4.0. Official Ukrainian documents were
considered and left out: their status for redistribution in a dataset was not clear enough.

## Export audit, 2026-10-22

The original manifest's code licence is the repository-level GitHub API label,
not an assertion that every third-party asset has the same licence. Preserve
embedded notices, vendored licences and font licences alongside the corpus.
The local candidate carries `paper/data/attributions/inventory.json`, original
table publisher metadata, the image attribution CSV and the original captions.

The collected `fre5h/transliteration-go` tree did not contain its LICENSE file.
A separately retrieved MIT notice is pinned to revision
`48a04429ca36e49cf109808fda13e679b06a99c1`; all four collected files match that
revision byte for byte. It is a supplement, not a claim that the original
snapshot contained that file. `tts_uk` has a README MIT statement and three
preserved dependency licence files, rather than a separate author LICENSE.
`uaTranslit` contains both MIT LICENSE and application TERMS.md; its Rubik font
declares SIL OFL 1.1 in its own name table. The exported inventory keeps these
scope questions visible for review before redistribution.
