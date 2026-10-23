"""LLM relevance judges (spec section 4): the prompt, grade parsing, and the local Gemma 4 judge via llama-server.

Gemini provides primary initial grades and local Gemma 4 12B provides independent initial grades for sensitivity.
They share a model provider family. Disagreements receive a within-model Gemini review under the same guideline;
Gemma E4B is ancillary and GPT is comparison only (see the dated judging protocol).
"""

from __future__ import annotations

import json
import re
import urllib.request

GUIDELINE = (
    "Оціни, наскільки документ відповідає на запит людини, яка шукає у власному архіві.\n"
    "2 — документ прямо відповідає на запит або це саме той документ, який шукають.\n"
    "1 — документ частково стосується запиту: та сама тема, але не відповідь.\n"
    "0 — документ не стосується запиту.\n"
    "Відповідай лише однією цифрою: 0, 1 або 2."
)


def judge_prompt(query: str, snippet: str) -> str:
    return f"{GUIDELINE}\n\nЗапит: {query}\n\nДокумент:\n{snippet}\n\nОцінка:"


def parse_grade(text: str) -> int | None:
    match = re.search(r"(?<!\d)([012])(?!\d)", text)
    return int(match.group(1)) if match else None


def judge_with_llama(query: str, snippet: str, url: str = "http://127.0.0.1:8282") -> int | None:
    # enable_thinking=False: Gemma 4 otherwise writes its reasoning into the answer and the grade never arrives
    body = json.dumps({"messages": [{"role": "user", "content": judge_prompt(query, snippet)}],
                       "temperature": 0, "max_tokens": 8,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        content = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
    return parse_grade(content)


def judge_with_ollama(query: str, snippet: str, model: str = "gemma4:12b",
                      url: str = "http://127.0.0.1:11434") -> int | None:
    """Independent initial sensitivity grades from local Gemma 4 served by Ollama, greedy decoding."""
    # think=False: Gemma 4 otherwise spends the whole token budget on hidden reasoning and returns no grade
    # keep_alive: without it the server here unloads the model after every request (about 13 s per grade)
    body = json.dumps({"model": model, "stream": False, "think": False, "keep_alive": "30m",
                       # num_ctx 2048: a prompt is under 1 000 tokens; the default 16k cache pushes the 12B model
                       # partly onto the CPU on an 8 GB GPU (17 s per grade)
                       "options": {"temperature": 0, "num_predict": 8, "num_ctx": 2048},
                       "messages": [{"role": "user", "content": judge_prompt(query, snippet)}]}).encode("utf-8")
    request = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        content = json.loads(response.read().decode("utf-8"))["message"]["content"]
    return parse_grade(content)


AGY_MODEL = "gemini-3.8-flash-medium"
BATCH_SCHEMA = {
    "type": "object",
    "properties": {"grades": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "grade": {"type": "integer"}},  # 0-2 checked by parse_batch
        "required": ["id", "grade"]}}},
    "required": ["grades"],
}


def batch_prompt(pairs: list[tuple[str, str, str]]) -> str:
    """pairs: (id, query, snippet). Every pair is graded on its own under the same guideline."""
    items = "\n\n".join(f"[{pid}]\nЗапит: {query}\nДокумент:\n{snippet}" for pid, query, snippet in pairs)
    return (f"{GUIDELINE.rsplit(chr(10), 1)[0]}\n"
            "Кожну пару оцінюй окремо, незалежно від інших пар.\n"
            'Відповідай лише JSON: {"grades": [{"id": "<номер у дужках>", "grade": 0|1|2}, ...]} '
            "з оцінкою для кожної пари.\n\n" + items)


def parse_batch(text: str, ids: list[str]) -> dict[str, int]:
    """Grades 0-2 for the asked ids; anything else (bad JSON, other ids, other values) is dropped, never guessed."""
    text = text.strip()
    if text.startswith("```"):  # Gemini wraps the JSON in a markdown fence despite the schema
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        grades = json.loads(text)["grades"]
    except (ValueError, KeyError, TypeError):
        return {}
    asked = set(ids)
    out, seen, duplicates = {}, set(), set()
    for item in grades if isinstance(grades, list) else []:
        if not isinstance(item, dict) or str(item.get("id")) not in asked:
            continue
        pid = str(item["id"])
        if pid in seen:
            duplicates.add(pid)
        seen.add(pid)
        if type(item.get("grade")) is int and item["grade"] in (0, 1, 2):
            out[pid] = item["grade"]
    return {pid: grade for pid, grade in out.items() if pid not in duplicates}


def judge_batch_with_agy(pairs: list[tuple[str, str, str]], model: str = AGY_MODEL, runner=None,
                         timeout_s: int = 600, prompt_builder=None) -> tuple[dict[str, int], dict]:
    """Gemini through the Antigravity CLI (agy) in print mode, sandboxed, in an empty directory, with a JSON schema.

    Returns the grades and a record of the call (model, conversation id, token usage, raw response) for the log.
    """
    import subprocess
    import tempfile

    runner = runner or subprocess.run
    with tempfile.TemporaryDirectory() as empty:
        argv = ["agy", "-p", (prompt_builder or batch_prompt)(pairs), "--model", model, "--sandbox", "--disable-slash-commands",
                "--output-format", "json", "--json-schema", json.dumps(BATCH_SCHEMA, ensure_ascii=False),
                "--print-timeout", f"{timeout_s}s"]
        done = runner(argv, cwd=empty, capture_output=True, text=True, timeout=timeout_s + 60)
    record = {"model": model, "returncode": done.returncode, "stderr": done.stderr[-2000:]}
    try:
        reply = json.loads(done.stdout)
    except ValueError:
        record["stdout"] = done.stdout[-2000:]
        return {}, record
    record.update({k: reply.get(k) for k in ("conversation_id", "status", "usage", "duration_seconds")})
    record["response"] = reply.get("response", "")
    return parse_batch(record["response"] or "", [pid for pid, _, _ in pairs]), record


CODEX_MODEL = "gpt-6-luna"
CODEX_EFFORT = "xhigh"


def _strict(schema: dict) -> dict:
    """OpenAI structured output wants additionalProperties: false on every object."""
    if schema.get("type") == "object":
        schema = {**schema, "additionalProperties": False,
                  "properties": {k: _strict(v) for k, v in schema["properties"].items()}}
    elif schema.get("type") == "array":
        schema = {**schema, "items": _strict(schema["items"])}
    return schema


def judge_batch_with_codex(pairs: list[tuple[str, str, str]], model: str = CODEX_MODEL, effort: str = CODEX_EFFORT,
                           runner=None, timeout_s: int = 900) -> tuple[dict[str, int], dict]:
    """GPT through the Codex CLI (codex exec), read-only sandbox, ephemeral session, in an empty directory.

    GPT-6-Luna also wrote the UA-PAR queries, so this judge grades its own queries: it is reported for comparison and
    does not enter the qrels.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    runner = runner or subprocess.run
    with tempfile.TemporaryDirectory() as tmp:
        schema, last, work = Path(tmp) / "schema.json", Path(tmp) / "last.txt", Path(tmp) / "work"
        work.mkdir()
        schema.write_text(json.dumps(_strict(BATCH_SCHEMA)), encoding="utf-8")
        argv = ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-s", "read-only",
                "--skip-git-repo-check", "--ephemeral", "--output-schema", str(schema), "-o", str(last),
                "-C", str(work), "--json", "-"]
        done = runner(argv, input=batch_prompt(pairs), capture_output=True, text=True, timeout=timeout_s)
        response = last.read_text(encoding="utf-8") if last.exists() else ""
    record = {"model": f"{model}/{effort}", "returncode": done.returncode, "stderr": done.stderr[-2000:],
              "status": "SUCCESS" if done.returncode == 0 and response else "ERROR", "response": response}
    for line in done.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "thread.started":
            record["conversation_id"] = event.get("thread_id")
        elif event.get("type") == "turn.completed":
            record["usage"] = event.get("usage")
    return parse_batch(response, [pid for pid, _, _ in pairs]), record
