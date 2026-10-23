"""Collect the API-based UA-PAR sources into raw_dir (network; build.py then lays them out offline).

- Wikipedia uk/ru: articles found by topic search, full plain text, one article per request (the API returns whole-article
  extracts one at a time), with a descriptive User-Agent as the Wikimedia policy asks.
- data.gov.ua: CSV datasets whose CKAN licence is open (see allowed_dataset).
- GitHub: repositories with topic 'ukrainian' under MIT or Apache-2.0, source tarball of the default branch.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "UA-PAR-research/0.1 (archive-index benchmark; https://github.com/cyberdid/archive-index-public)"
OPEN_DATASET_LICENCES = {"cc-by", "cc-zero", "odc-by", "odc-pddl", "cc-by-sa"}
OPEN_REPO_LICENCES = {"MIT", "Apache-2.0"}
MIN_ARTICLE_CHARS = 3000
MAX_TABLE_BYTES = 5 * 1024 * 1024
MAX_REPO_KB = 20000

# (slug, Ukrainian search, Russian search): topics of a technical research archive plus general knowledge.
TOPICS = [
    ("radio", "радіозв'язок", "радиосвязь"), ("antennas", "антена", "антенна"),
    ("sdr", "програмно-визначене радіо", "программно-определяемая радиосистема"),
    ("lte", "LTE мобільний зв'язок", "LTE мобильная связь"), ("satellites", "супутниковий зв'язок", "спутниковая связь"),
    ("signal", "цифрова обробка сигналів", "цифровая обработка сигналов"),
    ("electronics", "електроніка мікросхема", "электроника микросхема"),
    ("microcontrollers", "мікроконтролер", "микроконтроллер"), ("uav", "безпілотний літальний апарат", "беспилотный летательный аппарат"),
    ("ml", "машинне навчання", "машинное обучение"), ("neural", "штучна нейронна мережа", "искусственная нейронная сеть"),
    ("linux", "Linux операційна система", "Linux операционная система"),
    ("programming", "мова програмування", "язык программирования"), ("crypto", "криптографія", "криптография"),
    ("security", "інформаційна безпека", "информационная безопасность"), ("networks", "комп'ютерна мережа", "компьютерная сеть"),
    ("physics", "електромагнітне випромінювання", "электромагнитное излучение"), ("math", "теорія ймовірностей", "теория вероятностей"),
    ("history", "історія України", "история Украины"), ("literature", "українська література", "украинская литература"),
    ("geography", "географія України", "география Украины"), ("biology", "біологія клітини", "биология клетки"),
    ("medicine", "медицина діагностика", "медицина диагностика"), ("economics", "економіка ринок", "экономика рынок"),
    ("music", "музика композитор", "музыка композитор"),
]


def _get_json(url: str, params: dict | None = None) -> dict:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code not in (429, 503) or attempt == 7:
                raise
            # honour Retry-After; otherwise back off exponentially (Wikimedia rate limits bursts)
            retry = (error.headers or {}).get("Retry-After")
            time.sleep(float(retry) if retry and str(retry).isdigit() else min(60.0, 2.0 ** attempt))
    raise RuntimeError("unreachable")


def keep_article(text: str, title: str) -> bool:
    """Long, real articles only: no stubs, lists or disambiguation pages."""
    lowered = title.lower()
    if lowered.startswith(("список", "перелік")) or "(значення)" in lowered or "(значения)" in lowered:
        return False
    return len(text) >= MIN_ARTICLE_CHARS


def allowed_dataset(package: dict) -> bool:
    return package.get("license_id") in OPEN_DATASET_LICENCES


def allowed_repo(repo: dict) -> bool:
    licence = repo.get("license") or {}
    return licence.get("spdx_id") in OPEN_REPO_LICENCES


def _wiki_page(api: str, pageid: int, get_json) -> dict:
    pages = get_json(api, {"action": "query", "pageids": pageid, "prop": "extracts|info", "explaintext": 1,
                           "inprop": "url", "format": "json"})["query"]["pages"]
    return next(iter(pages.values()))


def collect_wiki(lang: str, per_topic: int, raw_dir: Path, pause: float = 0.05, get_json=None, topics=None,
                 workers: int = 2, max_hits: int = 2000) -> int:
    """Save up to per_topic long articles for every topic as raw_dir/wiki_<lang>/<pageid>.json. Returns the count.

    Search results are paged (500 per request, up to max_hits per topic); article extracts are fetched by a few
    parallel workers, since the API returns a whole-article extract one page at a time.
    """
    from concurrent.futures import ThreadPoolExecutor

    get_json = get_json or _get_json
    api = f"https://{lang}.wikipedia.org/w/api.php"
    out = raw_dir / f"wiki_{lang}"
    out.mkdir(parents=True, exist_ok=True)
    seen = {int(p.stem) for p in out.glob("*.json")}
    topic_of = {}
    for f in out.glob("*.json"):
        topic_of.setdefault(json.loads(f.read_text(encoding="utf-8")).get("topic"), set()).add(int(f.stem))
    saved = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for slug, uk_query, ru_query in (topics or TOPICS):
            query = uk_query if lang == "uk" else ru_query
            taken = len(topic_of.get(slug, ()))
            offset = 0
            while taken < per_topic and offset < max_hits:
                result = get_json(api, {"action": "query", "list": "search", "srsearch": query, "srlimit": 500,
                                        "sroffset": offset, "srnamespace": 0, "format": "json"})
                hits = [h for h in result["query"]["search"]
                        if h["pageid"] not in seen and h.get("size", 0) >= MIN_ARTICLE_CHARS]
                for start in range(0, len(hits), workers * 4):
                    if taken >= per_topic:
                        break
                    batch = hits[start:start + workers * 4]
                    for page in pool.map(lambda h: _wiki_page(api, h["pageid"], get_json), batch):
                        text = page.get("extract", "")
                        if taken >= per_topic or page["pageid"] in seen or not keep_article(text, page["title"]):
                            continue
                        record = {"pageid": page["pageid"], "title": page["title"], "url": page["fullurl"],
                                  "text": text, "topic": slug}
                        (out / f"{page['pageid']}.json").write_text(json.dumps(record, ensure_ascii=False),
                                                                     encoding="utf-8")
                        seen.add(page["pageid"])
                        taken += 1
                        saved += 1
                    time.sleep(pause)
                if "continue" not in result:
                    break
                offset = result["continue"]["sroffset"]
    return saved


def collect_datagovua(count: int, raw_dir: Path) -> int:
    """Save count open-licence CSV datasets as raw_dir/datagovua/<id>.csv plus <id>.json metadata."""
    out = raw_dir / "datagovua"
    out.mkdir(parents=True, exist_ok=True)
    saved = len(list(out.glob("*.csv")))
    start = 0
    while saved < count:
        result = _get_json("https://data.gov.ua/api/3/action/package_search",
                           {"q": "res_format:CSV", "rows": 100, "start": start, "sort": "metadata_modified desc"})["result"]
        if not result["results"]:
            break
        for package in result["results"]:
            if saved >= count or not allowed_dataset(package) or (out / f"{package['id']}.csv").exists():
                continue
            resource = next((r for r in package.get("resources", []) if (r.get("format") or "").upper() == "CSV"), None)
            if resource is None or (resource.get("size") or 0) > MAX_TABLE_BYTES:
                continue
            try:
                request = urllib.request.Request(resource["url"], headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read(MAX_TABLE_BYTES + 1)
            except Exception:
                continue
            if len(data) > MAX_TABLE_BYTES or len(data) < 50:
                continue
            (out / f"{package['id']}.csv").write_bytes(data)
            meta = {"id": package["id"], "title": package.get("title", ""), "licence": package["license_id"],
                    "url": f"https://data.gov.ua/dataset/{package['id']}",
                    "publisher": (package.get("organization") or {}).get("title", "")}
            (out / f"{package['id']}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            saved += 1
        start += 100
    return saved


def collect_github(count: int, raw_dir: Path) -> int:
    """Save count MIT/Apache-2.0 repositories with topic 'ukrainian' as raw_dir/github/<owner>__<repo>/."""
    out = raw_dir / "github"
    out.mkdir(parents=True, exist_ok=True)
    saved = len([p for p in out.iterdir() if p.is_dir()])
    for licence in ("mit", "apache-2.0"):
        result = _get_json("https://api.github.com/search/repositories",
                           {"q": f"topic:ukrainian license:{licence} size:<{MAX_REPO_KB}", "sort": "stars", "per_page": 50})
        for repo in result.get("items", []):
            target = out / repo["full_name"].replace("/", "__")
            if saved >= count or target.exists() or not allowed_repo(repo):
                continue
            url = f"https://codeload.github.com/{repo['full_name']}/tar.gz/HEAD"
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                for member in tar.getmembers():
                    parts = Path(member.name).parts[1:]  # drop the "<repo>-<sha>/" prefix
                    if not member.isfile() or not parts or ".." in parts:
                        continue
                    dest = target.joinpath(*parts)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tar.extractfile(member).read())
            meta = {"full_name": repo["full_name"], "url": repo["html_url"], "licence": repo["license"]["spdx_id"]}
            (target / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            saved += 1
    return saved


def main() -> int:
    raw = Path("paper/data/raw")
    print("wiki uk", collect_wiki("uk", 92, raw), flush=True)
    print("wiki ru", collect_wiki("ru", 46, raw), flush=True)
    print("datagovua", collect_datagovua(180, raw), flush=True)
    print("github", collect_github(30, raw), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
