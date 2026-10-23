from paper.corpus.collect import TOPICS, allowed_dataset, allowed_repo, keep_article


def test_only_open_dataset_licences_are_taken():
    assert allowed_dataset({"license_id": "cc-by"}) and allowed_dataset({"license_id": "cc-zero"})
    assert not allowed_dataset({"license_id": "notspecified"})
    assert not allowed_dataset({"license_id": "other-at"})


def test_only_mit_and_apache_repositories_are_taken():
    assert allowed_repo({"license": {"spdx_id": "MIT"}}) and allowed_repo({"license": {"spdx_id": "Apache-2.0"}})
    assert not allowed_repo({"license": {"spdx_id": "GPL-3.0"}}) and not allowed_repo({"license": None})


def test_short_and_list_articles_are_dropped():
    assert keep_article("Стаття " * 600, title="Антена")
    assert not keep_article("коротко", title="Антена")
    assert not keep_article("Стаття " * 600, title="Список антен")
    assert not keep_article("Стаття " * 600, title="Антена (значення)")


def test_topics_have_ascii_slugs_and_both_languages():
    assert len(TOPICS) >= 20
    for slug, uk, ru in TOPICS:
        assert slug.isascii() and slug == slug.lower() and uk and ru


def test_wiki_search_pages_past_the_first_100_hits(tmp_path):
    from paper.corpus.collect import collect_wiki

    calls = []

    def fake_get_json(url, params):
        calls.append(params)
        if params.get("list") == "search":
            offset = params.get("sroffset", 0)
            hits = [{"pageid": offset + i, "size": 9000} for i in range(params["srlimit"]) if offset + i < 700]
            more = {"continue": {"sroffset": offset + len(hits)}} if offset + len(hits) < 700 else {}
            return {"query": {"search": hits}, **more}
        pid = params["pageids"]
        return {"query": {"pages": {str(pid): {"pageid": pid, "title": f"Стаття {pid}", "fullurl": f"u/{pid}",
                                                "extract": "слово " * 700}}}}

    import paper.corpus.collect as collect

    saved = collect_wiki("uk", 600, tmp_path, pause=0, get_json=fake_get_json, topics=collect.TOPICS[:1], workers=3)
    assert saved == 600 and len(list((tmp_path / "wiki_uk").glob("*.json"))) == 600
    assert any(p.get("sroffset") for p in calls if p.get("list") == "search")


def test_get_json_waits_and_retries_on_429(monkeypatch):
    import io
    import urllib.error

    import paper.corpus.collect as collect

    calls, sleeps = [], []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=0):
        calls.append(request.full_url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "7"}, None)
        return Response(b'{"ok": 1}')

    monkeypatch.setattr(collect.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(collect.time, "sleep", sleeps.append)
    assert collect._get_json("https://uk.wikipedia.org/w/api.php", {"a": 1}) == {"ok": 1}
    assert len(calls) == 3 and sleeps[:2] == [7.0, 7.0]
