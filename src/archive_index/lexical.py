"""Lexical terms for FTS5: lowercase tokens plus Ukrainian/Russian lemmas.

Lemmas let a query like "точку доступу" match a document that says "точка доступу".
pymorphy3 can return several parses for one word; every normal form is kept, which
favours recall. Ranking is left to BM25.
"""

from __future__ import annotations

import functools
import re

import pymorphy3

_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*")
# Identifiers: numbers or codes joined by dots, slashes or hyphens (36.101, RVZ0014A, 19.6.0, a/b-2).
_IDENT = re.compile(r"[^\W_]+(?:[./][^\W_]+)+|[^\W_]+(?:-[0-9][^\W_]*)+")
_UK_ONLY = set("іїєґ")
_MIN_LEN = 2


@functools.lru_cache(maxsize=None)
def _analyzer(lang: str) -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer(lang=lang)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text) if len(token) >= _MIN_LEN]


@functools.lru_cache(maxsize=200_000)
def lemmas(token: str) -> frozenset[str]:
    """Normal forms of a Cyrillic token (uk and ru analyzers). Latin tokens return empty."""
    if not any("Ѐ" <= ch <= "ӿ" for ch in token):
        return frozenset()
    languages = ["uk"] if any(ch in _UK_ONLY for ch in token) else ["uk", "ru"]
    forms: set[str] = set()
    for lang in languages:
        try:
            analyzer = _analyzer(lang)
        except Exception:  # missing dictionary package for a language
            continue
        for parse in analyzer.parse(token):
            forms.add(parse.normal_form.lower())
    return frozenset(forms)


def identifiers(text: str) -> list[str]:
    """Identifiers such as 36.101 or 19.6.0, lowercased, in order of appearance."""
    return [m.group(0).lower() for m in _IDENT.finditer(text) if len(m.group(0)) >= 3]


def index_terms(text: str) -> str:
    """Space-separated terms stored in the FTS5 `terms` column."""
    terms: list[str] = []
    for token in tokenize(text):
        terms.append(token)
        terms.extend(sorted(lemmas(token)))
    terms.extend(identifiers(text))
    return " ".join(terms)


def query_expression(query: str) -> str:
    """FTS5 MATCH expression.

    Identifiers in the query (36.101) are matched as exact phrases and their parts are not used on
    their own, so "36.109" never matches a document that only contains "36". If additional words
    accompany the identifier, the identifier clause is AND-ed with the OR-group of the other words.
    """
    idents = identifiers(query)
    ident_clauses = [f'("{ident}")' for ident in idents]
    stripped = _IDENT.sub(" ", query) if idents else query
    word_clauses: list[str] = []
    for token in tokenize(stripped):
        forms = {token} | set(lemmas(token))
        word_clauses.append("(" + " OR ".join(f'"{form}"' for form in sorted(forms)) + ")")
    if ident_clauses and word_clauses:
        return "(" + " OR ".join(ident_clauses) + ") AND (" + " OR ".join(word_clauses) + ")"
    if ident_clauses:
        return " OR ".join(ident_clauses)
    return " OR ".join(word_clauses)
