"""Booktabs fragments; numerical values are identical in English and Ukrainian."""
from __future__ import annotations

METRICS = ('nDCG@10', 'R@100', 'RR@10', 'Success@1', 'Success@5')
CATEGORIES = ('known_item', 'topical', 'technical', 'cross_lingual')
SYSTEMS = ('bm25_plain', 'bm25_lemma', 'dense_eg2', 'dense_e5', 'rrf', 'adaptive',
           'adaptive-phrase', 'adaptive-ident', 'adaptive-short', 'dense_eg2_256')
ABLATIONS = ('adaptive', 'adaptive-phrase', 'adaptive-ident', 'adaptive-short', 'dense_eg2_256')
VARIANTS = ('gemini_initial', 'gemma12_initial', 'gemini_reviewed')
LABELS = {
    'en': {'system': 'System', 'known_item': 'Known item', 'topical': 'Topical', 'technical': 'Technical',
           'cross_lingual': r'Cross-lingual', 'gemini_initial': 'Gemini initial', 'gemma12_initial': 'Gemma12 initial',
           'gemini_reviewed': r'Gemini reviewed', 'language': 'Language', 'image': 'Image', 'speech': 'Speech',
           'mode': 'Mode', 'query': 'Queries', 'items': 'Items', 'uk': 'Ukrainian', 'ru': 'Russian'},
    'uk': {'system': 'Система', 'known_item': 'Відомий об’єкт', 'topical': 'Тематичні', 'technical': 'Технічні',
           'cross_lingual': 'Міжмовні', 'gemini_initial': 'Gemini початкові', 'gemma12_initial': 'Gemma12 початкові',
           'gemini_reviewed': 'Gemini переглянуті', 'language': 'Мова', 'image': 'Зображення', 'speech': 'Мовлення',
           'mode': 'Режим', 'query': 'Запити', 'items': 'Об’єкти', 'uk': 'Українська', 'ru': 'Російська'},
}


def tex_escape(text) -> str:
    escape = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
              '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(escape.get(c, c) for c in str(text))


def _table(headers, rows, comment=''):
    columns = len(headers) - 1
    return ('% ' + comment.replace('\n', '\n% ') + '\n'
            r'\begingroup\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.3}' + '\n'
            r'\noindent\begin{tabularx}{\linewidth}{@{}l*{' + str(columns) + r'}{>{\centering\arraybackslash}X}@{}}' + '\n'
            r'\toprule' + '\n' + ' & '.join(headers) + r' \\' + '\n' + r'\midrule' + '\n' +
            '\n'.join(' & '.join(row) + r' \\' for row in rows) + '\n' + r'\bottomrule' + '\n' +
            r'\end{tabularx}\endgroup' + '\n')


def _cell(stat, significant=False):
    marker = r'\textsuperscript{\textdagger}' if significant else ''
    return (r'\shortstack{' + f"{stat['mean']:.3f}" + marker + r' \\ ' +
            r'{\footnotesize[' + f"{stat['ci_low']:.3f}, {stat['ci_high']:.3f}" + ']}}')


def _systems(report, names=None):
    if names is not None: return [s for s in names if s in report['systems']]
    # Source systems are retrieval algorithms, never judge variants. Exclude GPT-labelled extras defensively.
    return [s for s in SYSTEMS if s in report['systems']] + sorted(
        s for s in report['systems'] if s not in SYSTEMS and 'gpt' not in s.lower())


def _metric_header(metric):
    return tex_escape(metric) + (r'\textsuperscript{pool}' if metric == 'R@100' else '')


def _significant(report, system, metric):
    reference = report.get('provenance', {}).get('reference_system', 'adaptive')
    return reference == 'adaptive' and system != 'adaptive' and any(
        row['system'] == system and row.get('p_holm', 1) < .05 for row in report.get('comparisons', {}).get(metric, []))


def main_table(results, lang='en', systems=None) -> str:
    """Two panels preserve all five metrics and readable confidence bounds at 160 mm."""
    text = LABELS[lang]
    note = ('Means with 95% bootstrap CI; n=' + str(results['queries']) + '. '
            'R@100 is pool-relative under incomplete judgments. '
            r'\(\dagger\): paired comparison against adaptive only, \(p_{\mathrm{Holm}}<0.05\).')
    out = []
    for metrics in (METRICS[:3], METRICS[3:]):
        rows = [[tex_escape(system)] + [_cell(results['systems'][system]['all'][metric],
                                              _significant(results, system, metric)) for metric in metrics]
                for system in _systems(results, systems)]
        out.append(_table([text['system']] + [_metric_header(m) for m in metrics], rows, note))
    # One common legend for both panels; comments above retain figure/table integration guidance.
    pool_label = 'pool-relative' if lang == 'en' else 'відносно пулу'
    legend = (r'{\footnotesize \textdagger: p\textsubscript{Holm}<0.05, adaptive; '
              'R@100: ' + pool_label + '.}' + '\n')
    # Avoid repeating the legend symbol in panel comments; only data cells and common legend use it.
    return '\n\\par\\medskip\n'.join(out).replace(r'\(\dagger\): paired', 'Significance: paired') + '\n\\par\\smallskip\n' + legend


def ablation_table(results, lang='en') -> str:
    return main_table(results, lang, ABLATIONS)


def _stratified(results, group, keys, lang):
    text = LABELS[lang]
    headers = [text['system']] + [tex_escape(text.get(k, k)) for k in keys]
    rows = [[tex_escape(s)] + [_cell(results['systems'][s][group][key]['nDCG@10']) for key in keys]
            for s in _systems(results)]
    return _table(headers, rows, f"nDCG@10; source subgroup 95% bootstrap CIs; primary={results.get('variant', 'unspecified')}.")


def category_table(results, lang='en') -> str:
    keys = [k for k in CATEGORIES if k in next(iter(results['systems'].values()))['by_category']]
    # Two columns per panel leave room for names and full subgroup intervals.
    return '\n\\par\\medskip\n'.join(_stratified(results, 'by_category', keys[i:i+2], lang)
                                       for i in range(0, len(keys), 2))


def language_table(results, lang='en') -> str:
    keys = sorted(next(iter(results['systems'].values()))['by_language'])
    return _stratified(results, 'by_language', keys, lang)


def sensitivity_table(reports, lang='en') -> str:
    variants = [v for v in VARIANTS if v in reports]
    first = reports[variants[0]]
    rows = [[tex_escape(s)] + [_cell(reports[v]['systems'][s]['all']['nDCG@10']) for v in variants]
            for s in _systems(first)]
    return _table([LABELS[lang]['system']] + [LABELS[lang][v] for v in variants], rows,
                  'nDCG@10 with 95% bootstrap CI. Gemini reviewed is a within-model review. '
                  'Gemini/Gemma12 share a provider family. No GPT or E4B in this principal comparison.')


def multimodal_table(results, lang='en') -> str:
    text = LABELS[lang]
    rows = []
    for language, values in sorted(results.get('images', {}).items()):
        rows.append((f"{text['image']} / {language}", values))
    for direction, modes in sorted(results.get('speech', {}).items()):
        for mode, values in sorted(modes.items()):
            mode_label = {'audio': 'аудіо', 'text': 'текст', 'both': 'обидва'}.get(mode, mode) if lang == 'uk' else mode
            rows.append((f"{text['speech']} / {direction} / {mode_label}", values))
    # Split query/item counts from metrics to avoid a wide table.
    metrics = ('recall@1', 'recall@5', 'recall@10', 'mrr')
    counts = _table([text['mode'], text['query'], text['items']],
                    [[tex_escape(label), str(v['queries']), str(v['items'])] for label, v in rows],
                    'Separate multimodal tasks; confidence intervals unavailable in source (not fabricated).')
    scores = _table([text['mode']] + [tex_escape(m) for m in metrics],
                    [[tex_escape(label)] + [f'{v[m]:.3f}' for m in metrics] for label, v in rows],
                    'Point estimates only; confidence intervals unavailable. Recall is task-specific.')
    return counts + '\n\\par\\medskip\n' + scores


def quantization_table(results, lang='en') -> str:
    headers = ['GGUF', 'Mean cosine', 'Worst cosine', 'MB', 'CPU s/text'] if lang == 'en' else [
        'GGUF', 'Середній косинус', 'Найменший косинус', 'МБ', 'CPU с/текст']
    rows = [[tex_escape(name), f"{v['mean']:.5f}", f"{v['worst']:.5f}", f"{v['file_mb']:.1f}",
             f"{v['cpu_s_per_text']:.3f}"] for name, v in results.items()]
    return _table(headers, rows, 'External quantization reference: embedding cosine agreement, not retrieval quality. '
                  'Point estimates; no CI available. Prompt counts: ' + ', '.join(f'{k}={v["n"]}' for k,v in results.items()))


def corpus_table(results, lang='en') -> str:
    rows = [['Manifest files' if lang == 'en' else 'Зібрані файли', str(results['manifest_files'])],
            ['Queries (all)' if lang == 'en' else 'Запити (усі)', str(results['queries'])]]
    rows += [[tex_escape(f'{name} index documents' if lang == 'en' else f'Документи індексу {name}'), str(v['documents'])]
             for name,v in sorted(results.get('indexes', {}).items())]
    for name, values in sorted(results.get('indexes', {}).items()):
        if 'vectors' in values:
            for key, en, uk in (('chunks', 'raw chunks', 'усі фрагменти'),
                                ('vectors', 'vectors', 'вектори'),
                                ('text_collection_vectors', 'text-collection vectors', 'вектори текстових колекцій')):
                if key not in values:
                    continue
                rows.append([tex_escape(f'{name}: {en if lang == "en" else uk}'), str(values[key])])
    return _table(['Population', 'Count'] if lang == 'en' else ['Сукупність', 'Кількість'], rows,
                  'Collected files and indexed documents are different populations; '
                  'E5 text-only and full indexes do not share the same document universe.')
