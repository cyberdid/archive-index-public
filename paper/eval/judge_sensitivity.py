"""Reproducible per-judge retrieval reports and ranking sensitivity from frozen inputs.

Use without --review for initial sensitivity, or supply a completely graded review to
include the Gemini within-model variant. This command never writes canonical qrels.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from paper.eval.metrics import DEFAULT_MEASURES, per_query
from paper.eval.report_main import build_report
from paper.eval.stats import bootstrap_ci, randomization_test
from paper.judge.agreement import cohen_kappa, disagreements
from paper.judge.finalize import (AGY_MODEL, REVIEW_KIND, Snapshot, add_snapshot_arguments, load_snapshot,
                                  merged_variants, sha256, write_json)

TIE_TOLERANCE = 1e-12
CAVEATS = {
    'pool': 'Depth-10 union of all frozen systems. Unjudged documents outside this pool are treated as nonrelevant '
            'by the metric evaluator; judgments are incomplete beyond the pool.',
    'R@100': 'R@100 is pool-relative recall under incomplete judgments, not corpus-wide recall.',
    'judge_independence': 'Gemini and Gemma12 provide independent initial grading but share the same provider '
                          '(Google model family); the Gemini review is within-model, not a third independent judge.',
    'ancillary_judges': 'Gemma E4B is ancillary; GPT is comparison only because GPT authored the queries. '
                        'Neither enters canonical qrels.',
    'no_answer': 'No-answer queries remain in qrels and are reported separately; they are excluded from '
                 'answerable-query retrieval averages.',
    'zero_relevant': 'Answerable queries with no positive grade in the judged pool score zero and remain '
                     'in the retrieval average.',
    'candidate_universe': 'The E5 run uses a text-only index, while the full index also includes media. '
                          'The runs do not share an equal candidate universe. Consult corpus_summary.json '
                          'for this experiment for document and code-document counts.',
    'target_coverage': 'Target construction coverage is reported dynamically in target_diagnostics. '
                       'The corpus may contain modalities without authored targets. Target-grade diagnostics '
                       'use only targets actually present in the judged pool.',
    'ties': f'System scores within absolute tolerance {TIE_TOLERANCE:g} share a rank; alphabetical order '
            'within a tie has no ranking meaning.',
}


def _ranked_docs(run, qid, k):
    return [doc for doc, _ in sorted(run.get(qid, {}).items(), key=lambda item: (-item[1], item[0]))[:k]]


def pool_coverage(data: Snapshot) -> dict:
    out = {}
    for name, run in data.runs.items():
        out[name] = {}
        for k in (10, 100):
            pairs = [(q.qid, doc) for q in data.queries for doc in _ranked_docs(run, q.qid, k)]
            judged = sum(key in data.pool_keys for key in pairs)
            out[name][f'at_{k}'] = {'retrieved_pairs': len(pairs), 'judged_pairs': judged,
                                   'unjudged_pairs': len(pairs) - judged,
                                   'judged_fraction': judged / len(pairs) if pairs else None}
    return out


def _population(data, grades):
    zero = sorted(q.qid for q in data.queries if q.category != 'no_answer' and
                  not any(g > 0 for (qid, _), g in grades.items() if qid == q.qid))
    return {**data.provenance['query_population'], 'answerable_zero_relevant_count': len(zero),
            'answerable_zero_relevant_query_ids': zero, 'positive_relevance_threshold': 1}


def _non_answer(data, grades):
    qids = [q.qid for q in data.queries if q.category == 'no_answer']
    out = {'queries': len(qids), 'systems': {},
           'interpretation': 'Retrieval and positive-grade diagnostics for the designated no-answer queries; '
                             'these are separate from answerable retrieval quality.'}
    for name, run in data.runs.items():
        nonempty = sum(bool(run.get(qid)) for qid in qids)
        counts = {k: sum(any(grades.get((qid, doc), 0) > 0 for doc in _ranked_docs(run, qid, k))
                         for qid in qids) for k in (1, 10)}
        out['systems'][name] = {'empty_retrieval_count': len(qids) - nonempty,
                                'nonempty_retrieval_count': nonempty,
                                **{f'queries_with_judged_positive_at_{k}': value for k, value in counts.items()},
                                **{f'judged_positive_rate_at_{k}': value / len(qids) if qids else None
                                   for k, value in counts.items()}}
    return out


def ranking_groups(scores: dict[str, float]) -> list[list[str]]:
    groups = []
    previous = None
    for name in sorted(scores, key=lambda n: (-scores[n], n)):
        if previous is None or not math.isclose(scores[name], previous, rel_tol=0, abs_tol=TIE_TOLERANCE):
            groups.append([])
            previous = scores[name]
        groups[-1].append(name)
    return groups


def _relation(scores, a, b):
    if math.isclose(scores[a], scores[b], rel_tol=0, abs_tol=TIE_TOLERANCE): return 0
    return 1 if scores[a] > scores[b] else -1


def _relation_text(a, b, relation):
    return f'{a} = {b}' if relation == 0 else (f'{a} > {b}' if relation > 0 else f'{b} > {a}')


def ordering_changes(initial: dict, variant: dict) -> list[dict]:
    changes = []
    for a, b in itertools.combinations(sorted(initial), 2):
        before, after = _relation(initial, a, b), _relation(variant, a, b)
        if before != after:
            changes.append({'systems': [a, b], 'initial': _relation_text(a, b, before),
                            'variant': _relation_text(a, b, after),
                            'kind': 'tie broken' if before == 0 else ('tie introduced' if after == 0 else 'reversal')})
    return changes


def planned_rq1_contrast(scores: dict, qids: list[str]) -> dict:
    """One planned morphology contrast on the primary endpoint, separate from adaptive/Holm."""
    a = [scores['bm25_lemma'][q]['nDCG@10'] for q in qids]
    b = [scores['bm25_plain'][q]['nDCG@10'] for q in qids]
    mean, low, high = bootstrap_ci([x - y for x, y in zip(a, b)])
    return {'label': 'planned RQ1 contrast: BM25 lemma versus BM25 plain', 'measure': 'nDCG@10',
            'system': 'bm25_lemma', 'reference': 'bm25_plain', 'n': len(qids),
            'system_mean': sum(a) / len(a), 'reference_mean': sum(b) / len(b),
            'mean_difference': mean, 'ci_low': low, 'ci_high': high, 'p': randomization_test(a, b),
            'multiplicity': 'Single planned nDCG@10 contrast; reported separately from the adaptive/Holm family.'}


def _target_coverage(data, grades):
    parts = {row['doc_id']: row.get('part', 'unknown') for row in
             (json.loads(line) for line in (data.root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()
              if line.strip())}
    targets = [q for q in data.queries if q.category != 'no_answer' and q.target]
    pooled = [q for q in targets if (q.qid, q.target) in grades]
    counts = {str(grade): sum(grades[(q.qid, q.target)] == grade for q in pooled) for grade in (0, 1, 2)}
    return {'answerable_query_count': sum(q.category != 'no_answer' for q in data.queries),
            'target_total': len(targets), 'target_in_pool': len(pooled), 'target_outside_pool': len(targets) - len(pooled),
            'target_parts': {part: sum(parts.get(q.target, 'unknown') == part for q in targets)
                             for part in sorted({parts.get(q.target, 'unknown') for q in targets})},
            'pooled_target_grade_counts': counts, 'target_grade_denominator': len(pooled),
            'pooled_target_grade_2_share': counts['2'] / len(pooled) if pooled else None}


def build_sensitivity(data: Snapshot, review: Path | None = None, reference: str = 'adaptive') -> tuple[dict, dict]:
    variants = merged_variants(data, review)
    if reference not in data.runs:
        raise ValueError(f'unknown reference system {reference}')
    provenance = {**data.provenance, 'reference_system': reference, 'tie_absolute_tolerance': TIE_TOLERANCE,
                  'variant_models': {name: data.provenance['models'].get(name, AGY_MODEL + ' (within-model review)')
                                     for name in variants}}
    if review is not None:
        provenance.update(review_sha256=sha256(review), review_kind=REVIEW_KIND, review_model=AGY_MODEL)
    coverage = pool_coverage(data)
    reports = {}
    for variant, grades in variants.items():
        qrels = {}
        for (qid, doc), grade in grades.items(): qrels.setdefault(qid, {})[doc] = grade
        if set(qrels) != {q.qid for q in data.queries}:
            raise ValueError(f'{variant}: qrels do not cover every query')
        report = build_report(data.queries, qrels, data.runs, reference=reference)
        serialized_qrels = ''.join(f'{qid} 0 {doc} {grade}\n' for (qid, doc), grade in sorted(grades.items()))
        report.update(variant=variant, provenance={**provenance, 'qrels_sha256': hashlib.sha256(
                          serialized_qrels.encode('utf-8')).hexdigest()},
                      population=_population(data, grades), pool_coverage=coverage, caveats=CAVEATS,
                      non_answer_diagnostics=_non_answer(data, grades))
        report['target_diagnostics'] = _target_coverage(data, grades)
        report['planned_contrasts'] = {}
        if {'bm25_lemma', 'bm25_plain'} <= set(data.runs):
            judged = {q.qid: qrels[q.qid] for q in data.queries if q.category != 'no_answer'}
            contrast_scores = {name: per_query(judged, data.runs[name], measures=('nDCG@10',))
                               for name in ('bm25_lemma', 'bm25_plain')}
            report['planned_contrasts']['rq1_lemma_vs_plain'] = planned_rq1_contrast(contrast_scores, sorted(judged))
        reports[variant] = report
    means = {variant: {metric: {system: report['systems'][system]['all'][metric]['mean']
                               for system in data.runs} for metric in DEFAULT_MEASURES}
             for variant, report in reports.items()}
    initial = means['gemini_initial']
    sensitivity = {'provenance': provenance, 'caveats': CAVEATS, 'pool_coverage': coverage,
                   'review_status': 'complete within-model Gemini review' if review else
                                    'not supplied; no canonical qrels produced',
                   'initial_agreement': {'pairs': len(data.pool_keys),
                         'disagreements': len(disagreements(data.grades('gemini'), data.grades('gemma12'))),
                         'linear_weighted_cohen_kappa': cohen_kappa(data.grades('gemini'), data.grades('gemma12'))},
                   'means': means,
                   'rankings': {variant: {metric: ranking_groups(values) for metric, values in metrics.items()}
                                for variant, metrics in means.items()},
                   'ordering_changes': {variant: {metric: ordering_changes(initial[metric], values)
                                                  for metric, values in metrics.items()}
                                        for variant, metrics in means.items() if variant != 'gemini_initial'},
                   'target_diagnostics': {name: report['target_diagnostics'] for name, report in reports.items()},
                   'planned_contrasts': {name: report['planned_contrasts'] for name, report in reports.items()},
                   'within_judge_comparisons': {name: report['comparisons'] for name, report in reports.items()},
                   'populations': {name: report['population'] for name, report in reports.items()},
                   'non_answer_diagnostics': {name: report['non_answer_diagnostics'] for name, report in reports.items()}}
    return reports, sensitivity


def sensitivity_markdown(sensitivity) -> str:
    p = sensitivity['provenance']
    pop = p['query_population']
    lines = ['# Judge sensitivity', '', f"Status: {sensitivity['review_status']}.", '',
             f"Population: {pop['total']} total queries, {pop['answerable']} answerable queries in metric averages, "
             f"{pop['no_answer']} no-answer queries reported separately.", '',
             f"Seed: {p['seed']}; bootstrap resamples: {p['bootstrap_resamples']}; "
             f"paired randomization resamples: {p['randomization_resamples']}; reference: {p['reference_system']}.", '',
             'Per-judge JSON reports contain category and language summaries, bootstrap confidence intervals, '
             'paired differences, paired randomization p-values and Holm-adjusted p-values.', '',
             '## Ranking tables', '', 'Equal scores share a rank. Scores are means across answerable queries.', '']
    variants = list(sensitivity['means'])
    systems = [name for group in sensitivity['rankings']['gemini_initial']['nDCG@10'] for name in group]
    for metric in DEFAULT_MEASURES:
        lines += [f'### {metric}', '', '| System | ' + ' | '.join(variants) + ' |',
                  '| --- | ' + ' | '.join('---:' for _ in variants) + ' |']
        ranks = {}
        for variant in variants:
            ranks[variant] = {}
            rank = 1
            for group in sensitivity['rankings'][variant][metric]:
                for system in group: ranks[variant][system] = rank
                rank += len(group)
        for system in systems:
            cells = [f"{sensitivity['means'][v][metric][system]:.6f} (#{ranks[v][system]})" for v in variants]
            lines.append(f'| {system} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Changed ordering relative to Gemini initial', '']
    for variant, metrics in sensitivity['ordering_changes'].items():
        lines += [f'### {variant}', '']
        for metric, changes in metrics.items():
            lines.append(f"- {metric}: {len(changes)} changed pairwise orderings.")
            for change in changes:
                lines.append(f"  - {change['initial']} → {change['variant']} ({change['kind']}).")
        lines.append('')
    lines += ['## Queries with no relevant pooled document', '']
    for variant, population in sensitivity['populations'].items():
        qids = population['answerable_zero_relevant_query_ids']
        lines.append(f"- {variant}: {len(qids)}; " + (', '.join(qids) if qids else 'none') + '.')
    lines += ['', '## Planned RQ1 contrast', '']
    for variant, contrasts in sensitivity['planned_contrasts'].items():
        if contrasts:
            c = contrasts['rq1_lemma_vs_plain']
            lines.append(f"- {variant}: lemma − plain nDCG@10 = {c['mean_difference']:.6f}; "
                         f"95% paired bootstrap CI [{c['ci_low']:.6f}, {c['ci_high']:.6f}]; "
                         f"paired randomization p = {c['p']:.6f}. {c['multiplicity']}")
    lines += ['', '## Authored target coverage', '']
    for variant, d in sensitivity['target_diagnostics'].items():
        lines.append(f"- {variant}: target_total={d['target_total']}, target_in_pool={d['target_in_pool']}, "
                     f"target_outside_pool={d['target_outside_pool']}; pooled target grade counts "
                     f"{d['pooled_target_grade_counts']}. Target-grade denominator={d['target_grade_denominator']}.")
    lines += ['', '## No-answer diagnostics', '',
              'Each cell shows empty retrieval queries / queries with a judged positive result at 10.', '',
              '| System | ' + ' | '.join(variants) + ' |',
              '| --- | ' + ' | '.join('---:' for _ in variants) + ' |']
    for system in systems:
        cells = []
        for variant in variants:
            d = sensitivity['non_answer_diagnostics'][variant]['systems'][system]
            cells.append(f"{d['empty_retrieval_count']} / {d['queries_with_judged_positive_at_10']}")
        lines.append(f'| {system} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Pool coverage', '',
              '| System | Judged retrieved pairs at 10 | Judged retrieved pairs at 100 |', '| --- | ---: | ---: |']
    for system in systems:
        c = sensitivity['pool_coverage'][system]
        lines.append(f"| {system} | {c['at_10']['judged_pairs']}/{c['at_10']['retrieved_pairs']} | "
                     f"{c['at_100']['judged_pairs']}/{c['at_100']['retrieved_pairs']} |")
    lines += ['', '## Limits', ''] + [f'- {text}' for text in sensitivity['caveats'].values()]
    lines += ['', '## Provenance', '', 'Input SHA-256 hashes (paths relative to the frozen snapshot):', '',
              '| Input | SHA-256 |', '| --- | --- |']
    lines += [f'| {path} | `{digest}` |' for path, digest in sorted(p['input_sha256'].items())]
    if 'review_sha256' in p:
        lines += ['', f"Graded review SHA-256: `{p['review_sha256']}`."]
    lines += ['', 'Models:', ''] + [f'- {variant}: {model}' for variant, model in p['variant_models'].items()]
    return '\n'.join(lines) + '\n'


def write_sensitivity(reports: dict, sensitivity: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for variant, report in reports.items(): write_json(out / f'ua_par_{variant}.json', report)
    write_json(out / 'judge_sensitivity.json', sensitivity)
    (out / 'judge_sensitivity.md').write_text(sensitivity_markdown(sensitivity), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_snapshot_arguments(parser)
    parser.add_argument('--review', type=Path, help='fully graded review; omit for initial sensitivity only')
    parser.add_argument('--out', type=Path, default=Path('paper/results'))
    parser.add_argument('--reference', default='adaptive')
    args = parser.parse_args()
    data = load_snapshot(args.snapshot, allow_empty_queries=args.allow_empty_query)
    if args.out.resolve().is_relative_to(args.snapshot.resolve()):
        raise ValueError('results must be outside the frozen snapshot')
    reports, sensitivity = build_sensitivity(data, args.review, args.reference)
    write_sensitivity(reports, sensitivity, args.out)
    print(f'wrote {len(reports)} per-judge reports and sensitivity JSON/Markdown to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
