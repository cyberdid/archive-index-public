"""Paired bootstrap confidence intervals, paired randomization tests and Holm correction (spec section 6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_RESAMPLES = 10_000
SEED = 2026


def bootstrap_ci(diffs, n: int = N_RESAMPLES, seed: int = SEED, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, low, high) of the per-query differences, resampling queries with replacement."""
    d = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(d.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def randomization_test(a, b, n: int = N_RESAMPLES, seed: int = SEED) -> float:
    """Two-sided paired sign-flip test on the mean difference; p = (count + 1) / (n + 1)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n, len(d)))
    flipped = np.abs((signs * d).mean(axis=1))
    count = int(np.sum(flipped >= observed - 1e-12))
    return (count + 1) / (n + 1)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment, monotone, capped at 1."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


@dataclass(frozen=True)
class Comparison:
    system: str
    mean: float
    reference_mean: float
    diff: float
    ci_low: float
    ci_high: float
    p: float
    p_holm: float


def compare_all(per_system: dict[str, dict[str, float]], reference: str = "adaptive") -> list[Comparison]:
    """Every system against the reference on the same queries, with bootstrap CIs and Holm-adjusted p-values."""
    ref = per_system[reference]
    qids = sorted(ref)
    rows, pvalues = {}, {}
    for name, scores in per_system.items():
        if name == reference:
            continue
        a = np.array([scores.get(q, 0.0) for q in qids])
        b = np.array([ref[q] for q in qids])
        mean_diff, low, high = bootstrap_ci(a - b)
        pvalues[name] = randomization_test(a, b)
        rows[name] = (float(a.mean()), float(b.mean()), mean_diff, low, high)
    adjusted = holm(pvalues)
    return [Comparison(name, *rows[name][:2], *rows[name][2:], pvalues[name], adjusted[name]) for name in sorted(rows)]
