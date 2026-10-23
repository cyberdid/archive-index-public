import numpy as np
import pytest

from paper.judge.agreement import cohen_kappa, disagreements, merge_grades
from paper.judge.llm_judge import judge_prompt, parse_grade
from paper.judge.pool import build_pool, export_sheet, read_sheet, read_trec_run, write_qrels


def _runs():
    a = [("q1", [(f"a{i}", 20 - i) for i in range(15)])]
    b = [("q1", [(f"b{i}", 20 - i) for i in range(15)])]
    return {"sysA": a, "sysB": b}


def test_pool_contains_top10_of_every_system():
    pool = build_pool(_runs(), depth=10)
    assert set(pool["q1"]) == {f"a{i}" for i in range(10)} | {f"b{i}" for i in range(10)}


def test_sheet_hides_systems_and_is_shuffled(tmp_path):
    pool = build_pool(_runs(), depth=10)
    path = tmp_path / "sheet.tsv"
    export_sheet(pool, {"q1": "антена"}, {d: f"текст {d}" for d in pool["q1"]}, path, annotator="claude")
    text = path.read_text(encoding="utf-8")
    assert "sysA" not in text and "sysB" not in text
    lines = text.splitlines()
    assert lines[0] == "qid\tquery\tdoc_id\tsnippet\tgrade"
    docs = [line.split("\t")[2] for line in lines[1:]]
    assert sorted(docs) == sorted(pool["q1"]) and docs != sorted(docs)


def test_read_sheet_and_qrels(tmp_path):
    path = tmp_path / "s.tsv"
    path.write_text("qid\tquery\tdoc_id\tsnippet\tgrade\nq1\tа\td1\tx\t2\nq1\tа\td2\tx\t0\n", encoding="utf-8")
    grades = read_sheet(path)
    assert grades == {("q1", "d1"): 2, ("q1", "d2"): 0}
    qrels = tmp_path / "qrels.txt"
    write_qrels(qrels, grades)
    assert qrels.read_text(encoding="utf-8").splitlines() == ["q1 0 d1 2", "q1 0 d2 0"]


def test_kappa_known_value():
    assert cohen_kappa({("q", "a"): 0, ("q", "b"): 1, ("q", "c"): 2, ("q", "d"): 2},
                       {("q", "a"): 0, ("q", "b"): 1, ("q", "c"): 2, ("q", "d"): 1}) == pytest.approx(0.714, abs=0.01)


def test_disagreements_and_merge():
    a = {("q", "a"): 0, ("q", "b"): 2}
    b = {("q", "a"): 0, ("q", "b"): 1}
    assert disagreements(a, b) == [("q", "b")]
    assert merge_grades(a, b, third={("q", "b"): 1}) == {("q", "a"): 0, ("q", "b"): 1}


def test_trec_run_reader(tmp_path):
    path = tmp_path / "r.trec"
    path.write_text("q1 Q0 d1 1 3.0 s\nq1 Q0 d2 2 2.0 s\nq2 Q0 d3 1 1.0 s\n", encoding="utf-8")
    assert read_trec_run(path) == [("q1", [("d1", 3.0), ("d2", 2.0)]), ("q2", [("d3", 1.0)])]


def test_llm_grade_parsing_and_prompt():
    assert parse_grade("Оцінка: 2") == 2 and parse_grade("grade=1") == 1 and parse_grade("0") == 0
    assert parse_grade("не знаю") is None and parse_grade("5") is None
    prompt = judge_prompt("антена ретранслятора", "Антена на даху працює.")
    assert "антена ретранслятора" in prompt and "0" in prompt and "2" in prompt


def test_ollama_judge_sends_the_prompt_and_parses_the_grade(monkeypatch):
    import io
    import json

    from paper.judge import llm_judge

    sent = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=0):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data.decode("utf-8"))
        return Response(json.dumps({"message": {"content": "Оцінка: 2"}}).encode("utf-8"))

    monkeypatch.setattr(llm_judge.urllib.request, "urlopen", fake_urlopen)
    assert llm_judge.judge_with_ollama("антена", "Антена на даху.") == 2
    assert sent["url"].endswith("/api/chat") and sent["body"]["model"] == "gemma4:12b"
    assert sent["body"]["options"]["temperature"] == 0 and "антена" in sent["body"]["messages"][0]["content"]
    assert sent["body"]["think"] is False  # otherwise the grade never arrives (Gemma 4 reasons first)


def test_llama_judge_turns_thinking_off(monkeypatch):
    import io
    import json

    from paper.judge import llm_judge

    sent = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=0):
        sent["body"] = json.loads(request.data.decode("utf-8"))
        return Response(json.dumps({"choices": [{"message": {"content": "1"}}]}).encode("utf-8"))

    monkeypatch.setattr(llm_judge.urllib.request, "urlopen", fake_urlopen)
    assert llm_judge.judge_with_llama("антена", "Антена.") == 1
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False} and sent["body"]["temperature"] == 0


def test_snippets_are_title_and_first_chunk_by_corpus_id(tmp_path):
    from archive_index.indexer import run_index
    from archive_index.store import Store
    from paper.judge.sheets import SNIPPET_CHARS, snippets_from_index

    class TextFake:
        def documents(self, texts, titles=None):
            return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)

    root = tmp_path / "ua-par"
    (root / "radio" / "notes").mkdir(parents=True)
    body = "\n\n".join(f"Абзац {i} про антену ретранслятора і кабель до приймача на даху." * 4 for i in range(40))
    (root / "radio" / "notes" / "uk-1.md").write_text("# Антена\n\n" + body, encoding="utf-8")
    (root / "radio" / "notes" / "uk-2.md").write_text(
        "# Кабель\n\nДжерело: https://uk.wikipedia.org/wiki/x\n\n## Опис\n\n" + body, encoding="utf-8")
    store = Store(tmp_path / "full.sqlite3")
    run_index(store, TextFake(), projects_root=root, home=tmp_path / "h", collections=("wiki",))
    store.close()
    snippets = snippets_from_index(tmp_path / "full.sqlite3", root)
    assert sorted(snippets) == ["radio/notes/uk-1.md", "radio/notes/uk-2.md"]
    title, lead = snippets["radio/notes/uk-1.md"].split("\n", 1)
    assert title == "Антена" and lead.startswith("Абзац 0") and "Абзац 39" not in lead
    # A heading-only first chunk is followed by the next chunks, so the judge sees the text itself.
    assert "Абзац 0" in snippets["radio/notes/uk-2.md"]
    assert len(snippets["radio/notes/uk-2.md"]) >= SNIPPET_CHARS


def test_judge_sheet_fills_grades_and_resumes(tmp_path):
    from paper.judge.sheets import judge_sheet

    sheet = tmp_path / "secondary.tsv"
    sheet.write_text("qid\tquery\tdoc_id\tsnippet\tgrade\n"
                     "q1\tантена\td1\tантена на даху\t\nq1\tантена\td2\tкабель\t\nq2\tкіт\td3\tсобака\t\n",
                     encoding="utf-8")
    calls = []

    def judge(query, snippet):
        calls.append(snippet)
        if len(calls) == 2:
            raise KeyboardInterrupt  # the run stops in the middle
        return 2 if query in snippet else 0

    with pytest.raises(KeyboardInterrupt):
        judge_sheet(sheet, judge)
    assert read_sheet(sheet) == {("q1", "d1"): 2}
    judge_sheet(sheet, judge)
    assert read_sheet(sheet) == {("q1", "d1"): 2, ("q1", "d2"): 0, ("q2", "d3"): 0}
    assert calls == ["антена на даху", "кабель", "кабель", "собака"]


def test_judge_sheet_leaves_unparsed_answers_empty(tmp_path):
    from paper.judge.sheets import judge_sheet

    sheet = tmp_path / "s.tsv"
    sheet.write_text("qid\tquery\tdoc_id\tsnippet\tgrade\nq1\tа\td1\tx\t\n", encoding="utf-8")
    report = judge_sheet(sheet, lambda q, s: None)
    assert read_sheet(sheet) == {} and report == {"graded": 0, "unparsed": 1, "already": 0}


def test_speech_snippets_carry_the_human_transcript():
    from paper.judge.sheets import add_transcripts

    snippets = {"recordings/uk/1.wav": "1.wav\naudio: 1.wav at 0-9s", "radio/notes/uk-1.md": "Антена\nтекст"}
    manifest = [{"doc_id": "recordings/uk/1.wav", "part": "speech", "transcript": "антена на даху"},
                {"doc_id": "radio/notes/uk-1.md", "part": "notes"}]
    out = add_transcripts(snippets, manifest)
    assert out["recordings/uk/1.wav"] == "1.wav\naudio: 1.wav at 0-9s\nСказано в записі: антена на даху"
    assert out["radio/notes/uk-1.md"] == "Антена\nтекст"


def test_snippets_show_percent_encoded_links_as_text():
    from paper.judge.sheets import readable

    assert readable("Джерело: https://uk.wikipedia.org/wiki/%D0%90%D0%BD%D1%82%D0%B5%D0%BD%D0%B0 текст 100%") \
        == "Джерело: https://uk.wikipedia.org/wiki/Антена текст 100%"


def test_batch_prompt_numbers_every_pair_under_the_guideline():
    from paper.judge.llm_judge import GUIDELINE, batch_prompt

    text = batch_prompt([("1", "антена", "Антена на даху"), ("2", "кабель", "Коаксіальний кабель")])
    assert GUIDELINE.splitlines()[0] in text
    assert "[1]" in text and "[2]" in text and "Антена на даху" in text and "Коаксіальний кабель" in text


def test_parse_batch_keeps_only_valid_grades_of_asked_ids():
    from paper.judge.llm_judge import parse_batch

    text = '{"grades": [{"id": "1", "grade": 2}, {"id": "2", "grade": 5}, {"id": "9", "grade": 1}, {"id": "3"}]}'
    assert parse_batch(text, ["1", "2", "3"]) == {"1": 2}
    assert parse_batch("not json", ["1"]) == {}
    fenced = '```json\n{"grades": [{"id": "1", "grade": 1}]}\n```'
    assert parse_batch(fenced, ["1"]) == {"1": 1}  # Gemini wraps the JSON in a markdown fence


def test_agy_judge_runs_gemini_in_a_sandbox_and_keeps_the_record():
    import json

    from paper.judge.llm_judge import judge_batch_with_agy

    seen = {}

    def runner(argv, **kwargs):
        seen["argv"] = argv
        response = json.dumps({"grades": [{"id": "1", "grade": 2}, {"id": "2", "grade": 0}]})
        out = json.dumps({"conversation_id": "c1", "status": "SUCCESS", "response": response,
                          "usage": {"input_tokens": 100, "output_tokens": 20}})
        return type("Done", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    grades, record = judge_batch_with_agy([("1", "антена", "Антена"), ("2", "кіт", "Собака")], runner=runner)
    assert grades == {"1": 2, "2": 0}
    argv = seen["argv"]
    assert argv[argv.index("--model") + 1] == "gemini-3.8-flash-medium"
    assert "--sandbox" in argv and argv[argv.index("--output-format") + 1] == "json"
    assert record["conversation_id"] == "c1" and record["model"] == "gemini-3.8-flash-medium"
    assert record["usage"] == {"input_tokens": 100, "output_tokens": 20} and "response" in record


def test_batched_judging_fills_the_sheet_resumes_and_logs_every_call(tmp_path):
    import json

    from paper.judge.sheets import judge_sheet_batched

    sheet = tmp_path / "gemini.tsv"
    sheet.write_text("qid\tquery\tdoc_id\tsnippet\tgrade\n" + "".join(
        f"q1\tантена\td{i}\tтекст {i}\t\n" for i in range(5)), encoding="utf-8")
    log = tmp_path / "gemini.calls.jsonl"
    batches = []

    def judge(pairs):
        batches.append([pid for pid, _, _ in pairs])
        if len(batches) == 2:
            raise KeyboardInterrupt
        return {pid: 1 for pid, _, _ in pairs}, {"conversation_id": f"c{len(batches)}"}

    with pytest.raises(KeyboardInterrupt):
        judge_sheet_batched(sheet, judge, log, batch_size=2)
    assert read_sheet(sheet) == {("q1", "d0"): 1, ("q1", "d1"): 1}
    report = judge_sheet_batched(sheet, judge, log, batch_size=2)
    assert len(read_sheet(sheet)) == 5 and report["graded"] == 3
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [c["conversation_id"] for c in calls] == ["c1", "c3", "c4"]
    assert calls[1]["pairs"] == [["q1", "d2"], ["q1", "d3"]]


def test_make_sheets_never_overwrites_a_graded_sheet(tmp_path):
    from paper.judge.sheets import refuse_graded

    sheet = tmp_path / "gemini.tsv"
    sheet.write_text("qid\tquery\tdoc_id\tsnippet\tgrade\nq1\tа\td1\tx\t2\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        refuse_graded(sheet)
    sheet.write_text("qid\tquery\tdoc_id\tsnippet\tgrade\nq1\tа\td1\tx\t\n", encoding="utf-8")
    refuse_graded(sheet)  # an empty sheet may be rewritten
    refuse_graded(tmp_path / "missing.tsv")


def test_batched_judging_stops_at_a_failed_call(tmp_path):
    """A spent quota fails every call: the run stops at the first one instead of running through the sheet."""
    import json

    from paper.judge.sheets import judge_sheet_batched

    sheet = tmp_path / "gemini.tsv"
    sheet.write_text("qid\tquery\tdoc_id\tsnippet\tgrade\n" + "".join(
        f"q1\tантена\td{i}\tтекст {i}\t\n" for i in range(6)), encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    calls = []

    def judge(pairs):
        calls.append(pairs)
        if len(calls) == 1:
            return {pid: 1 for pid, _, _ in pairs}, {"status": "SUCCESS"}
        return {}, {"status": "ERROR", "stderr": "RESOURCE_EXHAUSTED"}

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        judge_sheet_batched(sheet, judge, log, batch_size=2)
    assert len(calls) == 2 and len(read_sheet(sheet)) == 2
    assert [json.loads(line)["status"] for line in log.read_text(encoding="utf-8").splitlines()] == ["SUCCESS", "ERROR"]


def test_codex_judge_runs_read_only_with_a_schema_and_keeps_the_record():
    import json
    from pathlib import Path

    from paper.judge.llm_judge import judge_batch_with_codex

    seen = {}

    def runner(argv, **kwargs):
        seen["argv"], seen["input"] = argv, kwargs.get("input")
        out = Path(argv[argv.index("-o") + 1])
        assert json.loads(Path(argv[argv.index("--output-schema") + 1]).read_text())["required"] == ["grades"]
        out.write_text(json.dumps({"grades": [{"id": "1", "grade": 2}, {"id": "2", "grade": 0}]}))
        events = [{"type": "thread.started", "thread_id": "t1"},
                  {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 9}}]
        return type("Done", (), {"returncode": 0, "stdout": "\n".join(map(json.dumps, events)), "stderr": ""})()

    grades, record = judge_batch_with_codex([("1", "антена", "Антена"), ("2", "кіт", "Собака")], runner=runner)
    assert grades == {"1": 2, "2": 0}
    argv = seen["argv"]
    assert argv[:2] == ["codex", "exec"] and argv[argv.index("-m") + 1] == "gpt-6-luna"
    assert 'model_reasoning_effort="xhigh"' in argv and argv[argv.index("-s") + 1] == "read-only"
    assert "--ephemeral" in argv and "Антена" in seen["input"]
    assert record["conversation_id"] == "t1" and record["status"] == "SUCCESS"
    assert record["usage"] == {"input_tokens": 100, "output_tokens": 9} and record["model"] == "gpt-6-luna/xhigh"
