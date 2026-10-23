# How UA-PAR queries are written

Author of the queries: GPT-6-Luna (OpenAI) in the Codex CLI, under these rules (the study is done without other
people; see the spec, section 4). The model differs from the primary relevance judge (Claude), so no judge grades
its own queries. Each query is written
from what a person remembers about a document (its title, topic, a fact), never by copying its text.

## Categories and quotas

| Category | Quota | What it is | Example |
|---|---|---|---|
| `known_item` | 40 | a specific document the person saw before; `target` names it | «стаття про те, як працює супергетеродинний приймач» |
| `topical` | 60 | an information need that several documents may answer; `target` optional | «чому LTE використовує OFDM замість CDMA» |
| `technical` | 40 | identifiers, model numbers, standards, numbers in tables | «3GPP TS 36.211 фізичний рівень», «ESP32-S3 струм споживання» |
| `cross_lingual` | 30 | the query is in another language than the target (uk ↔ ru, en → uk/ru) | «antenna gain measurement» → a Ukrainian note |
| `no_answer` | 30 | a plausible question the corpus does not cover; must give zero word matches | «правила гри в керлінг на льоду Антарктиди» |

## Languages

At least 40 % of all queries in Ukrainian and at least 20 % in Russian; the rest English. `lang` is the language of the
query text.

## Rules

1. Do not copy. A topical or cross-lingual query may share at most 5 consecutive tokens with its target document; the
   validator rejects longer runs.
2. Write like a person searching their own archive: short (3–10 words), sometimes imprecise, sometimes with a typo-free
   inflected form (запит «антени» for a document about «антена»).
3. Known-item queries describe the document, they do not quote its title verbatim.
4. Technical queries contain the identifier as people write it (with dots, hyphens, versions).
5. No-answer queries must be checked against the index (`validate(..., store=...)`) and replaced if they match anything.
6. Every query gets a unique `qid` (`q001` …), `author` naming the model that wrote it (`gpt-6-luna` for UA-PAR v1), and `target` when the category needs one.

File: `paper/data/queries.tsv`, tab-separated with the header `qid text lang category author target`.
