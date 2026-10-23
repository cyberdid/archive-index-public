import pytest
from archive_index.chunking import chunk_markdown, pack_paragraphs
from archive_index.lexical import index_terms, query_expression, tokenize
from archive_index.redact import redact


def test_redact_removes_known_token_shapes_and_counts_kinds():
    # Fake values assembled at run time so no secret-shaped literal sits in the source.
    fake_openai = "sk-" + "a" * 24
    fake_aws = "AKIA" + "B" * 16
    fake_github = "ghp_" + "c" * 24
    text = f"key {fake_openai} and {fake_aws} and {fake_github}"
    clean, counts = redact(text)
    assert fake_openai not in clean
    assert fake_aws not in clean
    assert fake_github not in clean
    assert counts["openai_key"] == 1
    assert counts["aws_access_key"] == 1
    assert counts["github_token"] == 1


def test_redact_credential_assignment_keeps_the_name_drops_the_value():
    clean, counts = redact("API_KEY=supersecretvalue123 and password: hunter2hunter")
    assert "supersecretvalue123" not in clean
    assert "hunter2hunter" not in clean
    assert "API_KEY=" in clean
    assert counts["credential_assignment"] == 2


def test_redact_leaves_ordinary_prose_alone():
    text = "Ми обговорили token count і password reset flow без значень."
    clean, counts = redact(text)
    assert clean == text
    assert sum(counts.values()) == 0


def test_redact_private_key_block():
    header = "-----BEGIN RSA " + "PRIVATE KEY-----"
    footer = "-----END RSA " + "PRIVATE KEY-----"
    block = f"{header}\nMIIBOgIBAAJBAK\n{footer}"
    clean, counts = redact("before\n" + block + "\nafter")
    assert "MIIBOgIBAAJBAK" not in clean  # key body removed
    assert "before" in clean and "after" in clean
    assert counts["private_key"] == 1


def test_pack_paragraphs_respects_limit_and_keeps_paragraphs_whole():
    text = "\n\n".join(["абзац один " * 5, "абзац два " * 5, "абзац три " * 5])
    chunks = pack_paragraphs(text, max_chars=120)
    assert all(len(c) <= 120 for c in chunks)
    joined = "\n\n".join(chunks)
    assert "абзац один" in joined and "абзац три" in joined


def test_pack_paragraphs_hard_splits_oversized_paragraph():
    chunks = pack_paragraphs("x" * 2500, max_chars=1000)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_chunk_markdown_keeps_heading_per_chunk():
    md = "intro line\n\n# Setup\n\nInstall the thing.\n\n## Usage\n\nRun it."
    pairs = chunk_markdown(md, max_chars=500)
    headings = [h for h, _ in pairs]
    assert "" in headings  # preamble
    assert "Setup" in headings and "Usage" in headings
    assert any("Install the thing." in body for h, body in pairs if h == "Setup")


def test_tokenize_lowercases_and_drops_single_letters():
    assert tokenize("A Wi-Fi точку доступу") == ["wi-fi", "точку", "доступу"]


def test_lemmas_match_inflected_ukrainian_forms():
    terms = index_terms("точка доступу налаштовує")
    query = query_expression("точку доступу")
    # The query must mention the lemma that the indexed text also produced.
    assert "точка" in terms.split()
    assert '"точка"' in query


def test_lemmas_ignore_guesses_of_the_other_language():
    from archive_index.lexical import lemmas

    # The Russian analyzer does not know these Ukrainian words and invents verbs for them.
    assert lemmas("навчанню") == {"навчання"}
    assert lemmas("етапи") == {"етап"}
    # The Ukrainian analyzer invents a verb for a Russian word.
    assert lemmas("какую") == {"какой"}
    # Homonyms that a dictionary knows stay: recall over precision for real words.
    assert {"коли", "кола", "колоть"} <= lemmas("коли")


def test_lemmas_keep_guesses_when_no_dictionary_knows_the_word():
    from archive_index.lexical import lemmas

    assert lemmas("лорофікації")  # unknown to both dictionaries: the predictor's forms are kept


def test_query_expression_empty_for_no_tokens():
    assert query_expression("! ?") == ""


def test_query_expression_combines_identifier_and_words_with_and():
    expr = query_expression("36.101 антена")
    assert '"36.101"' in expr
    assert "AND" in expr
    assert "антена" in expr


def test_phrase_close_prefers_sequence_over_scattered_numbers():
    from archive_index.search import _phrase_close

    assert _phrase_close("192 168 169 tcp", "192.168.8.169 | tcp | 53")  # one extra token allowed
    assert not _phrase_close("192 168 169 tcp", "192.168.1.5 | udp | 53 | 169 tcp far away " + "x " * 9 + "192")
    assert not _phrase_close("одне", "одне")  # a single word is not a phrase


def test_ukrainian_apostrophe_variants_give_one_token():
    for word in ("м’ята", "мʼята", "м'ята", "м‘ята"):
        assert tokenize(word) == ["м'ята"]
    assert tokenize("п’ять") == ["п'ять"]


def test_query_and_index_agree_on_apostrophes():
    assert "м'ята" in index_terms("м’ята в саду").split()
    assert '"м\'ята"' in query_expression("мʼята")


def test_coverage_counts_whole_words_not_substrings(tmp_path):
    """The adaptive switch to exact-match mode must not fire on fragments: '36' is not in '1366', 'ip' not in 'script'."""
    from archive_index.search import _covers
    from archive_index.store import ChunkRecord, Store

    store = Store(tmp_path / "t.sqlite3")
    try:
        store.replace_document(collection="wiki", path="/p/a.md", size=1, mtime_ns=1, sha256="x", title="a",
                               project="p", chunks=[ChunkRecord(ord=0, heading="", locator="", occurred_at="",
                               body="Роздільна здатність 1366 точок, javascript у браузері", terms="", vector=None,
                               body_sha="b")])
        cid = store.conn.execute("SELECT id FROM chunks").fetchone()[0]
        assert _covers(store, "36 ip", cid) == 0.0
        assert _covers(store, "1366 javascript", cid) == 1.0
        assert _covers(store, "Роздільна здатність ip", cid) == pytest.approx(2 / 3)
        assert _covers(store, "м’ята", cid) == 0.0
    finally:
        store.close()
