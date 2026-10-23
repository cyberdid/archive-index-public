"""Agreement between two judges and the third-pass merge (plan Task 7, spec section 4)."""

from __future__ import annotations

GRADES = (0, 1, 2)


def cohen_kappa(a: dict, b: dict) -> float:
    """Linear-weighted Cohen's kappa on the pairs both judges graded (grades 0, 1, 2)."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan")
    n = len(keys)
    span = max(GRADES) - min(GRADES)
    weight = lambda i, j: 1 - abs(i - j) / span
    observed = sum(weight(a[k], b[k]) for k in keys) / n
    pa = {g: sum(1 for k in keys if a[k] == g) / n for g in GRADES}
    pb = {g: sum(1 for k in keys if b[k] == g) / n for g in GRADES}
    expected = sum(pa[i] * pb[j] * weight(i, j) for i in GRADES for j in GRADES)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def disagreements(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) & set(b) if a[k] != b[k])


def merge_grades(primary: dict, secondary: dict, third: dict) -> dict:
    """Final grade: the agreed grade, else the third pass; pairs graded by one judge only keep the primary grade."""
    final = {}
    for key in set(primary) | set(secondary):
        if key in primary and key in secondary and primary[key] != secondary[key]:
            final[key] = third[key]
        else:
            final[key] = primary.get(key, secondary.get(key))
    return final
