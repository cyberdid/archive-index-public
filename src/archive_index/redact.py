"""Secret redaction applied to every text unit before it reaches the index."""

from __future__ import annotations

import re
from collections import Counter

_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.S,
)

_TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
]

# key: value / key=value for common credential names. Only explicit separators,
# so ordinary prose such as "token count 12" is left alone.
_ASSIGNMENT = re.compile(
    r"(?i)\b((?:api[_ -]?key|api[_ -]?secret|secret|token|password|passwd|"
    r"client[_ -]?secret|access[_ -]?token|consumer[_ -]?key|consumer[_ -]?secret|"
    r"auth[_ -]?token)\b)(\s*[:=]\s*)(['\"`]?)([^\s'\"`,;]{6,})"
)


def redact(text: str) -> tuple[str, Counter]:
    """Return (clean_text, counts_by_kind). Values are never returned, only kinds."""
    counts: Counter = Counter()

    def _replace_key(match: re.Match[str]) -> str:
        counts["private_key"] += 1
        return "[REDACTED:private_key]"

    text = _PRIVATE_KEY.sub(_replace_key, text)

    for kind, pattern in _TOKEN_PATTERNS:
        def _replace(match: re.Match[str], kind: str = kind) -> str:
            counts[kind] += 1
            return f"[REDACTED:{kind}]"

        text = pattern.sub(_replace, text)

    def _replace_assignment(match: re.Match[str]) -> str:
        counts["credential_assignment"] += 1
        return f"{match.group(1)}{match.group(2)}{match.group(3)}[REDACTED]"

    text = _ASSIGNMENT.sub(_replace_assignment, text)
    return text, counts
