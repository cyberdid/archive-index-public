"""Generate publication fragments/figures from measured JSON; no model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

from paper.report.figures import figure_data, figure_names, render_figures
from paper.report.tables import (METRICS, SYSTEMS, VARIANTS, ablation_table, category_table, corpus_table,
                                 language_table, main_table, multimodal_table, quantization_table,
                                 sensitivity_table, tex_escape)

REQUIRED = ('ua_par_gemini_initial.json', 'ua_par_gemma12_initial.json', 'judge_sensitivity.json',
            'judges.json', 'multimodal.json', 'corpus_summary.json', 'quantization_reference.json')
PLANNED = ('efficiency.json', 'field_aggregate.json')
# Added after the audit of 2026-10-22; each table appears only when its measured source exists.
OPTIONAL = ('encoder_control.json', 'ann_scaling.json', 'ann_reference_audit.json', 'covers_fix.json', 'extension.json')
TABLE_NAMES = ('main', 'ablation', 'category', 'language', 'multimodal', 'judge_sensitivity',
               'corpus', 'quantization_reference', 'evidence_notes')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value, where):
    if isinstance(value, dict):
        for key, item in value.items(): _finite(item, f'{where}/{key}')
    elif isinstance(value, list):
        for i, item in enumerate(value): _finite(item, f'{where}/{i}')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{where}: nonfinite value')


def _stat(stat, where, expected_n=None):
    try:
        low, mean, high = (stat[key] for key in ('ci_low', 'mean', 'ci_high'))
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (low, mean, high)):
            raise ValueError('nonfinite or invalid interval')
        if not 0 <= low <= mean <= high <= 1:
            raise ValueError('bad confidence interval or score outside 0..1')
        if type(stat['n']) is not int or stat['n'] < 1 or (expected_n is not None and stat['n'] != expected_n):
            raise ValueError('sample count mismatch')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{where}: missing/bad score, CI, or count: {exc}') from exc


def load_sources(results: Path, primary=None):
    sources = {}
    for name in REQUIRED:
        if not (results / name).is_file(): raise ValueError(f'missing measured source {name}')
        sources[name] = json.loads((results / name).read_text(encoding='utf-8'))
    for name in ('ua_par_gemini_reviewed.json', *PLANNED, *OPTIONAL):
        if (results / name).is_file(): sources[name] = json.loads((results / name).read_text(encoding='utf-8'))
    for name, source in sources.items(): _finite(source, name)
    ann_audit = sources.get('ann_reference_audit.json', {})
    if ann_audit.get('complete'):
        ann = sources.get('ann_scaling.json', {})
        if (not ann.get('complete') or ann_audit['source_ann_sha256'] != digest(results / 'ann_scaling.json')
                or ann_audit['source_sha256'] != ann['source_sha256']
                or set(ann_audit['sizes']) != set(ann['sizes']) or ann_audit['queries'] != ann['queries']):
            raise ValueError('archive vector-path audit is not bound to the complete ANN experiment')
    primary = primary or ('gemini_reviewed' if 'ua_par_gemini_reviewed.json' in sources else 'gemini_initial')
    if primary not in ('gemini_initial', 'gemini_reviewed') or f'ua_par_{primary}.json' not in sources:
        raise ValueError(f'primary measured source unavailable: {primary}')
    reports = {variant: sources[f'ua_par_{variant}.json'] for variant in VARIANTS
               if f'ua_par_{variant}.json' in sources}
    try:
        n = reports[primary]['queries']
        for variant, report in reports.items():
            if report.get('variant') != variant or report['queries'] != n or type(n) is not int or n < 1:
                raise ValueError(f'{variant}: query count or variant mismatch')
            if set(report['systems']) != set(SYSTEMS):
                raise ValueError(f'{variant}: expected all ten frozen retrieval systems')
            for system, body in report['systems'].items():
                for metric in METRICS: _stat(body['all'][metric], f'{variant}/{system}/{metric}', n)
                for group in ('by_category', 'by_language'):
                    if not body[group]: raise ValueError(f'{variant}/{system}: empty {group}')
                    for key, measures in body[group].items():
                        for metric in METRICS: _stat(measures[metric], f'{variant}/{system}/{group}/{key}/{metric}')
                for metric in METRICS:
                    actual = sources['judge_sensitivity.json']['means'][variant][metric][system]
                    expected = body['all'][metric]['mean']
                    if actual != expected: raise ValueError(f'{variant}/{system}: sensitivity mean disagrees with source')
            for metric, comparisons in report['comparisons'].items():
                for row in comparisons:
                    if not 0 <= row['p_holm'] <= 1: raise ValueError('invalid Holm-adjusted p-value')
        multimodal = sources['multimodal.json']
        rows = list(multimodal.get('images', {}).values()) + [v for modes in multimodal.get('speech', {}).values()
                                                              for v in modes.values()]
        if not rows: raise ValueError('empty multimodal source')
        for row in rows:
            if any(not 0 <= row[m] <= 1 for m in ('recall@1', 'recall@5', 'recall@10', 'mrr')):
                raise ValueError('invalid multimodal score')
            if row['queries'] < 1 or row['items'] < 1: raise ValueError('invalid multimodal population')
        for name, row in sources['quantization_reference.json'].items():
            if not 0 <= row['worst'] <= row['mean'] <= 1 or any(row[k] <= 0 for k in ('n', 'file_mb', 'cpu_s_per_text')):
                raise ValueError(f'invalid quantization reference {name}')
    except (KeyError, TypeError) as exc:
        raise ValueError(f'incomplete measured source contract: {exc}') from exc
    return sources, reports, primary


def _evidence_notes(sources, reports, primary, lang):
    report = reports[primary]
    population = report.get('provenance', {}).get('query_population', {})
    status = 'Gemini within-model reviewed' if primary == 'gemini_reviewed' else 'Gemini initial; review unavailable or not selected'
    if lang == 'uk':
        uk_status = 'Gemini: перегляд тією самою моделлю' if primary == 'gemini_reviewed' else 'Gemini: початкові оцінки; перегляд не вибрано'
        lines = [f'Оцінювання: {uk_status}. Метрики: {report["queries"]} запитів із відповіддю; '
                 f'без відповіді: {population.get("no_answer", "не вказано")}.',
                 'R@100 відносно пулу; повної еталонної розмітки немає. Gemini та Gemma12 мають спільного розробника. '
                 'Перегляд Gemini виконується тією самою моделлю. GPT не включено до основних таблиць і рисунків.']
    else:
        lines = [f'Judging: {tex_escape(status)}. Metrics: {report["queries"]} answerable queries; '
                 f'no-answer queries: {population.get("no_answer", "not reported")}.',
                 'R@100 is pool-relative; judgments are incomplete, not full gold. Gemini and Gemma12 share a provider '
                 'family. Gemini review is within-model. GPT is excluded from principal tables and figures.']
    counts = sources['judges.json'].get('judges', {})
    for name in ('gemini', 'gemma4-12b'):
        if name in counts:
            label = 'graded pairs' if lang == 'en' else 'оцінених пар'
            lines.append(f'{tex_escape(name)}: {counts[name]["graded"]} {label}.')
    if 'initial_agreement' in sources['judge_sensitivity.json']:
        a = sources['judge_sensitivity.json']['initial_agreement']
        agreement_label = 'Initial disagreements' if lang == 'en' else 'Початкові розбіжності'
        kappa_label = 'linear-weighted Cohen kappa' if lang == 'en' else 'лінійно зважена каппа Коена'
        lines.append(f'{agreement_label}: {a["disagreements"]}/{a["pairs"]}; '
                     f'{kappa_label}: {a["linear_weighted_cohen_kappa"]:.3f}.')
    target = report.get('target_diagnostics')
    if target:
        if lang == 'en':
            lines.append(f'Targets: total={target["target_total"]}, in pool={target["target_in_pool"]}, '
                         f'outside pool={target["target_outside_pool"]}. Target-grade denominator={target["target_grade_denominator"]}.')
        else:
            lines.append(f'Цільові документи: всього={target["target_total"]}, у пулі={target["target_in_pool"]}, '
                         f'поза пулом={target["target_outside_pool"]}. Знаменник оцінок цільових документів={target["target_grade_denominator"]}.')
    for name in PLANNED:
        if name not in sources:
            label = 'Немає вимірювань; числових значень не наведено' if lang == 'uk' else 'No measured source; no numerical values reported'
            lines.append(f'{tex_escape(name)}: {label}.')
    if lang == 'en':
        lines += [tex_escape(caveat) for caveat in sources['corpus_summary.json'].get('caveats', [])]
    else:
        lines += ['Зібрані файли та індексовані документи є різними сукупностями.',
                  'Текстовий індекс E5 та повний індекс не мають однакового набору документів.']
        part_labels = {'notes': 'нотатки', 'images': 'зображення', 'speech': 'мовлення',
                       'code': 'код', 'tables': 'таблиці', 'documents': 'документи'}
        parts = sources['corpus_summary.json'].get('target_parts', {})
        if parts:
            lines.append('Типи цільових документів: ' + ', '.join(
                f'{part_labels.get(k,k)}={v}' for k,v in parts.items()) + '. Якість для решти типів не встановлено.')
    return '\\begingroup\\raggedright\n' + '\n\\par\n'.join(lines) + '\n\\par\\endgroup\n'


def table_artifacts(sources, reports, primary):
    report = reports[primary]
    out = {}
    for lang in ('en', 'uk'):
        fragments = {'main': main_table(report, lang), 'ablation': ablation_table(report, lang),
                     'category': category_table(report, lang), 'language': language_table(report, lang),
                     'multimodal': multimodal_table(sources['multimodal.json'], lang),
                     'judge_sensitivity': sensitivity_table(reports, lang),
                     'corpus': corpus_table(sources['corpus_summary.json'], lang),
                     'quantization_reference': quantization_table(sources['quantization_reference.json'], lang),
                     'evidence_notes': _evidence_notes(sources, reports, primary, lang)}
        fragments.update(optional_fragments(sources, lang))
        for name, text in fragments.items():
            out[f'{lang}/{name}.tex'] = f'% Generated from primary={primary}; see ../manifest.json for source SHA-256.\n' + text
    return out


def optional_fragments(sources, lang):
    from paper.report.extra_tables import (ann_table, control_table, field_table, gguf_table, index_table, judge_agreement_table,
                                           judge_checks_table, latency_table)
    from paper.report.extra_tables import SYSTEMS as COST_SYSTEMS

    out = {}
    judges = sources['judges.json']
    if judges.get('pairs') and all('distribution' in j for j in judges.get('judges', {}).values()):
        out['judge_checks'] = judge_checks_table(judges, lang)
        out['judge_agreement'] = judge_agreement_table(judges, lang)
    eff = sources.get('efficiency.json', {})
    if all(set(COST_SYSTEMS) <= set(eff.get('queries', {}).get(device, {})) for device in ('gpu', 'cpu')):
        out['latency'] = latency_table(eff, lang)
    if {'gpu', 'cpu'} <= set(eff.get('index', {})):
        out['indexing'] = index_table(eff, lang)
    if (eff.get('quantization', {}).get('retrieval', {}).get('quality')
            and eff['quantization']['retrieval'].get('complete', True)):
        out['gguf'] = gguf_table(eff['quantization']['retrieval'], lang)
    if 'ann_scaling.json' in sources and sources['ann_scaling.json'].get('complete', True):
        out['ann'] = ann_table(sources['ann_scaling.json'], lang, sources.get('ann_reference_audit.json'))
    if 'encoder_control.json' in sources:
        out['encoder_control'] = control_table(sources['encoder_control.json'], lang)
    if 'field_aggregate.json' in sources:
        out['field'] = field_table(sources['field_aggregate.json'], lang)
    if 'extension.json' in sources:
        from paper.report.tables import _cell, _table
        extension = sources['extension.json']
        variants = extension['variants']
        rows = [[tex_escape(system)] + [_cell(variants[v]['systems'][system]['all']['nDCG@10'])
                                       for v in ('gemini_initial', 'gemma12_initial')]
                for system in SYSTEMS if system in variants['gemini_initial']['systems']]
        out['extension'] = _table(['System' if lang == 'en' else 'Система', 'Gemini initial', 'Gemma 12B initial'],
                                  rows, 'nDCG@10, 95% query-bootstrap CI; separate 50-query extension; source extension.json')
        unique = extension.get('unique_wording_sensitivity')
        if unique:
            rows = [[tex_escape(system)] + [_cell(unique['variants'][v]['systems'][system]['all']['nDCG@10'])
                                           for v in ('gemini_initial', 'gemma12_initial')]
                    for system in ('bm25_plain', 'bm25_lemma', 'dense_e5', 'dense_eg2', 'adaptive')]
            heading = (f"Unique-wording sensitivity: {unique['queries']} queries." if lang == 'en' else
                       f"Чутливість на унікальних формулюваннях: {unique['queries']} запитів.")
            out['extensionunique'] = (tex_escape(heading) + '\n\\par\\smallskip\n'
                                     + _table(['System' if lang == 'en' else 'Система', 'Gemini initial', 'Gemma 12B initial'],
                                              rows, 'keep lowest qid per normalized wording; source extension.json'))
    # Bind conditional manuscript sections to this generation's evidence set, not
    # to potentially stale optional files left in the output directory.
    out['measurement_status'] = '\n'.join(
        r'\newif\ifpaper' + name + '\n' + r'\paper' + name + ('true' if name in out else 'false')
        for name in ('latency', 'indexing', 'gguf', 'ann', 'field', 'extension', 'extensionunique')) + '\n'
    return out


def unavailable_artifacts(sources):
    missing = []
    if 'efficiency.json' not in sources: missing.append('GPU efficiency table/latency-quality figure')
    else:
        eff = sources['efficiency.json']
        if not {'gpu', 'cpu'} <= set(eff.get('queries', {})): missing.append('complete CPU/GPU query latency')
        if not {'gpu', 'cpu'} <= set(eff.get('index', {})): missing.append('complete CPU/GPU indexing')
    quant = sources.get('efficiency.json', {}).get('quantization', {}).get('retrieval', {})
    if not quant.get('quality') or not quant.get('complete', True): missing.append('GGUF retrieval-quality comparison')
    if 'ann_scaling.json' not in sources or not sources['ann_scaling.json'].get('complete', True):
        missing.append('complete ANN scaling experiment')
    if 'extension.json' not in sources: missing.append('separate non-note query extension')
    if 'field_aggregate.json' not in sources: missing.append('private field study aggregates')
    return missing


def generate(results: Path, out: Path, primary=None):
    sources, reports, primary = load_sources(results, primary)
    if out.resolve() == results.resolve() or results.resolve().is_relative_to(out.resolve()):
        raise ValueError('output directory must not contain the source results directory')
    texts = table_artifacts(sources, reports, primary)
    chart_data = figure_data(reports, primary, sources.get('efficiency.json'), sources.get('ann_scaling.json'),
                             sources.get('ann_reference_audit.json'))
    for relative, text in texts.items():
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    render_figures(chart_data, out / 'figures')
    figure_files = [f'figures/{name}.{ext}' for name in figure_names(chart_data) for ext in ('pdf', 'png')]
    artifacts = sorted([*texts, *figure_files])
    manifest = {'schema': 1, 'primary': primary,
                'status': 'complete within-model Gemini review' if primary == 'gemini_reviewed' else
                          'initial judgments; review not available' if 'gemini_reviewed' not in reports else
                          'initial judgments explicitly selected; reviewed sensitivity also available',
                'software_versions': {'python': platform.python_version(), 'matplotlib': version('matplotlib'),
                                      'ir-measures': version('ir-measures'), 'numpy': version('numpy')},
                'source_sha256': {name: digest(results / name) for name in sorted(sources)},
                'missing_planned_sources': [name for name in PLANNED if name not in sources],
                'unavailable_planned_artifacts': unavailable_artifacts(sources),
                'query_population': reports[primary].get('provenance', {}).get('query_population', {}),
                'judging_provenance': reports[primary].get('provenance', {}), 'figure_data': chart_data,
                'artifact_sha256': {name: digest(out / name) for name in artifacts},
                'artifact_metadata': {'table_width': '160 mm linewidth; main/category split into panels',
                                      'table_intervals': '95% query bootstrap CIs from source; none fabricated',
                                      'figure_width_mm': 160, 'png_dpi': 300, 'pdf_fonttype': 42,
                                      'significance': 'paired p_holm<.05 against adaptive only',
                                      'R@100': 'pool-relative, incomplete judgments; not full gold'}}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2,
                                                allow_nan=False) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('paper/results'))
    parser.add_argument('--out', type=Path, default=Path('paper/manuscript/generated'))
    parser.add_argument('--primary', choices=('gemini_initial', 'gemini_reviewed'))
    args = parser.parse_args()
    manifest = generate(args.results, args.out, args.primary)
    print(f"Generated {len(manifest['artifact_sha256'])} artifacts; primary={manifest['primary']}; "
          f"unavailable={', '.join(manifest['unavailable_planned_artifacts']) or 'none'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
