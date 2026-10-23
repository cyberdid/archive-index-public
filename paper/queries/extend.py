"""UA-PAR extension: queries whose targets are documents, tables and code, not notes (audit of 2026-10-22).

The 200 frozen queries all target notes. These 50 extra queries are written by a model that is neither a relevance
judge nor the author of the frozen queries (Qwen 3.8 27B, local, through Ollama), from the beginning of each target
document, under the same rules: 3-10 words, written as a person who remembers the document, and no run of more than
five consecutive tokens shared with the document. They are evaluated as a separate population, never mixed into the
frozen 200.
"""

from __future__ import annotations

import json
import hashlib
import random
import urllib.request
from pathlib import Path

from paper.queries.schema import MAX_SHARED_RUN, Query, longest_shared_run, write_queries

COUNTS = {"documents": 20, "tables": 15, "code": 15}
CATEGORY = {"documents": "known_item", "tables": "known_item", "code": "technical"}
MIN_TEXT = 200
MIN_WORDS, MAX_WORDS = 3, 10
MODEL = "qwen3.8:27b-q4_K_M"
AUTHOR = "qwen3.8-27b"


def choose_targets(entries: list[dict], texts: dict[str, str], counts: dict[str, int] = COUNTS,
                   seed: int = 2029) -> list[dict]:
    rng = random.Random(seed)
    chosen = []
    for part, n in counts.items():
        ids = sorted(e["doc_id"] for e in entries if e["part"] == part and len(texts.get(e["doc_id"], "")) >= MIN_TEXT)
        lang = {e["doc_id"]: e.get("lang") for e in entries}
        for i, doc in enumerate(rng.sample(ids, min(n, len(ids)))):
            language = lang[doc] if lang[doc] in ("uk", "ru") else ("uk" if i % 2 == 0 else "ru")
            chosen.append({"doc_id": doc, "part": part, "lang": language, "category": CATEGORY[part]})
    return chosen


def query_prompt(title: str, text: str, lang: str, category: str) -> str:
    language = "українською" if lang == "uk" else "російською"
    kind = ("Людина пам'ятає, що в неї в архіві є цей документ, і шукає саме його."
            if category == "known_item" else
            "Людина шукає цей файл коду за тим, що він робить, або за назвами, які в ньому трапляються.")
    return (f"Ось початок документа з особистого архіву.\n\nНазва: {title}\n\n{text}\n\n"
            f"{kind} Напиши один пошуковий запит {language} мовою, 3-10 слів, так, як людина пише в пошук: "
            "коротко, своїми словами. Не цитуй назви дослівно і не копіюй з документа більше п'яти слів поспіль. "
            "Відповідай лише текстом запиту, без лапок і пояснень.")


def accept(query: str, document_text: str) -> bool:
    words = query.split()
    return (MIN_WORDS <= len(words) <= MAX_WORDS and "\n" not in query
            and longest_shared_run(query, document_text) <= MAX_SHARED_RUN)


def to_queries(rows: list[dict], start: int = 201, author: str = AUTHOR) -> list[Query]:
    return [Query(f"q{start + i:03d}", r["text"], r["lang"], r["category"], author, r["doc_id"]) for i, r in enumerate(rows)]


def _ollama(prompt: str, temperature: float, seed: int, url: str = "http://127.0.0.1:11434") -> str:
    body = json.dumps({"model": MODEL, "stream": False, "think": False, "keep_alive": "10m",
                       "options": {"temperature": temperature, "seed": seed, "num_predict": 64, "num_ctx": 4096},
                       "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    request = urllib.request.Request(url + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))["message"]["content"].strip().strip('"«»').strip()


def main() -> int:
    from paper.corpus.build import read_manifest
    from paper.judge.sheets import snippets_from_index

    corpus, out = Path("paper/data/ua-par"), Path("paper/data/queries-ext.tsv")
    if out.exists():
        raise FileExistsError(f"{out} exists; the extension is written once")
    entries = read_manifest(corpus / "manifest.jsonl")
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
        models = json.loads(response.read())["models"]
    model_digest = next(m["digest"] for m in models if m["name"] == MODEL)
    leads = snippets_from_index(Path("paper/data/indexes/full.sqlite3"), corpus)
    texts = {doc: lead.split("\n", 1)[-1] for doc, lead in leads.items()}
    rows, log = [], Path("paper/data/queries-ext.log.jsonl")
    targets = choose_targets(entries, texts)
    if len(targets) != sum(COUNTS.values()):
        raise ValueError("not enough eligible targets for the planned 50-query extension")
    previous = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    for target in targets:
        title = leads[target["doc_id"]].split("\n", 1)[0]
        prompt = query_prompt(title, texts[target["doc_id"]], target["lang"], target["category"])
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        completed = next((r for r in reversed(previous) if r["doc_id"] == target["doc_id"]
                          and r.get("accepted") and r.get("model") == MODEL
                          and r.get("model_digest") == model_digest
                          and r.get("prompt_sha256") == prompt_hash
                          and accept(r["response"], leads[target["doc_id"]])), None)
        if completed:
            rows.append({**target, "text": completed["response"]})
            print(target["doc_id"], "-> resumed", flush=True)
            continue
        for attempt, temperature in enumerate((0.0, 0.7, 0.7, 0.7)):
            text = _ollama(prompt, temperature, seed=attempt)
            ok = accept(text, leads[target["doc_id"]])
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"doc_id": target["doc_id"], "attempt": attempt, "temperature": temperature,
                                     "response": text, "accepted": ok, "model": MODEL, "model_digest": model_digest,
                                     "seed": attempt, "prompt_sha256": prompt_hash}, ensure_ascii=False) + "\n")
            if ok:
                rows.append({**target, "text": text})
                break
        print(target["doc_id"], "->", rows[-1]["text"] if rows and rows[-1]["doc_id"] == target["doc_id"] else "REJECTED",
              flush=True)
        if not rows or rows[-1]["doc_id"] != target["doc_id"]:
            raise ValueError(f"no valid query for {target['doc_id']}; no partial extension written")
    write_queries(out, to_queries(rows))
    print(f"wrote {out}: {len(rows)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
