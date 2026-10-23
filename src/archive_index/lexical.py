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


# Ukrainian is written with several apostrophe characters; all of them mean the same letter boundary.
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u02bc": "'", "\u2018": "'", "`": "'"})


def normalize_apostrophes(text: str) -> str:
    """Map ’ (U+2019), ʼ (U+02BC), ‘ (U+2018) and ` to the ASCII apostrophe, so «м’ята» stays one word."""
    return text.translate(_APOSTROPHES)


def tokenize(text: str) -> list[str]:
    text = normalize_apostrophes(text)
    return [token.lower() for token in _TOKEN.findall(text) if len(token) >= _MIN_LEN]


@functools.lru_cache(maxsize=200_000)
def lemmas(token: str) -> frozenset[str]:
    """Normal forms of a Cyrillic token (uk and ru analyzers). Latin tokens return empty."""
    if not any("Ѐ" <= ch <= "ӿ" for ch in token):
        return frozenset()
    languages = ["uk"] if any(ch in _UK_ONLY for ch in token) else ["uk", "ru"]
    known: set[str] = set()
    guessed: set[str] = set()
    for lang in languages:
        try:
            analyzer = _analyzer(lang)
        except Exception:  # missing dictionary package for a language
            continue
        for parse in analyzer.parse(token):
            (known if parse.is_known else guessed).add(parse.normal_form.lower())
    # A dictionary word keeps only dictionary forms: the other language's analyzer invents verbs
    # for it (навчанню -> навчаннить). Guesses are used only when no dictionary knows the word.
    return frozenset(known or guessed)


def identifiers(text: str) -> list[str]:
    """Identifiers such as 36.101 or 19.6.0, lowercased, in order of appearance."""
    return [m.group(0).lower() for m in _IDENT.finditer(text) if len(m.group(0)) >= 3]


def index_terms(text: str, lemmatize: bool = True) -> str:
    """Space-separated terms stored in the FTS5 `terms` column."""
    terms: list[str] = []
    for token in tokenize(text):
        terms.append(token)
        if lemmatize:
            terms.extend(sorted(lemmas(token)))
    terms.extend(identifiers(text))
    return " ".join(terms)


def query_expression(query: str, lemmatize: bool = True, identifiers: bool = True) -> str:
    """FTS5 MATCH expression.

    Identifiers in the query (36.101) are matched as exact phrases and their parts are not used on
    their own, so "36.109" never matches a document that only contains "36". If additional words
    accompany the identifier, the identifier clause is AND-ed with the OR-group of the other words.
    """
    query = normalize_apostrophes(query)
    idents = _identifiers(query) if identifiers else []
    ident_clauses = [f'("{ident}")' for ident in idents]
    stripped = _IDENT.sub(" ", query) if idents else query
    word_clauses: list[str] = []
    for token in tokenize(stripped):
        forms = {token} | (set(lemmas(token)) if lemmatize else set())
        word_clauses.append("(" + " OR ".join(f'"{form}"' for form in sorted(forms)) + ")")
    if ident_clauses and word_clauses:
        return "(" + " OR ".join(ident_clauses) + ") AND (" + " OR ".join(word_clauses) + ")"
    if ident_clauses:
        return " OR ".join(ident_clauses)
    return " OR ".join(word_clauses)


_identifiers = identifiers  # query_expression's parameter of the same name shadows the function
