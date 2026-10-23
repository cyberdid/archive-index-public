"""Standalone scientific PDF/PNG figures with source-backed asymmetric intervals."""
from __future__ import annotations

from pathlib import Path

from paper.report.tables import CATEGORIES, LABELS, SYSTEMS, VARIANTS

CATEGORY_SYSTEMS = ('bm25_lemma', 'dense_eg2', 'dense_e5', 'adaptive')
COLORS = ('#0072B2', '#009E73', '#D55E00', '#6F4C9B')
VARIANT_LABELS = {'gemini_initial': 'Gemini initial', 'gemma12_initial': 'Gemma12 initial',
                  'gemini_reviewed': 'Gemini reviewed (within model)'}


def figure_data(reports, primary, efficiency=None, ann=None, ann_reference=None):
    report = reports[primary]
    categories = [c for c in CATEGORIES if c in next(iter(report['systems'].values()))['by_category']]
    category = {'primary': primary, 'queries': report['queries'], 'categories': categories,
                'systems': [s for s in CATEGORY_SYSTEMS if s in report['systems']], 'rows': []}
    for category_name in categories:
        for system in category['systems']:
            category['rows'].append({'category': category_name, 'system': system,
                                     **report['systems'][system]['by_category'][category_name]['nDCG@10']})
    category['counts'] = {c: next(r['n'] for r in category['rows'] if r['category'] == c) for c in categories}
    variants = [v for v in VARIANTS if v in reports]
    sensitivity = {'variants': variants, 'systems': [s for s in SYSTEMS if s in report['systems']],
                   'queries': report['queries'], 'rows': []}
    for variant in variants:
        for system in sensitivity['systems']:
            sensitivity['rows'].append({'variant': variant, 'system': system,
                                        **reports[variant]['systems'][system]['all']['nDCG@10']})
    data = {'category': category, 'sensitivity': sensitivity,
            'architecture': {'sources': ['src/archive_index/indexer.py', 'src/archive_index/search.py',
                                         'src/archive_index/store.py', 'src/archive_index/embed.py',
                                         'paper/runs/systems.py'],
                             'scope': 'default local backend; document collapse is benchmark-only'}}
    if efficiency:
        from paper.report.extra_tables import SYSTEMS as COST_SYSTEMS
        timings = efficiency.get('queries', {})
        if all(set(COST_SYSTEMS) <= set(timings.get(d, {})) for d in ('gpu', 'cpu')):
            data['latency'] = {'systems': list(COST_SYSTEMS), 'interval': 'p50 to p95; not a confidence interval',
                               'rows': [{'device': d, 'system': s, **timings[d][s]['latency_ms']}
                                        for d in ('gpu', 'cpu') for s in COST_SYSTEMS]}
            data['cost_quality'] = {
                'primary': primary, 'quality_scope': 'frozen original runs, before whole-word coverage correction',
                'latency_scope': 'current implementation, after correction; embedding and chunk search only',
                'rows': [{'system': s, 'gpu_p50_ms': timings['gpu'][s]['latency_ms']['p50'],
                          **report['systems'][s]['all']['nDCG@10']} for s in COST_SYSTEMS]}
        quant = efficiency.get('quantization', {}).get('retrieval', {})
        if quant.get('complete') and primary in quant.get('quality', {}):
            quality = quant['quality'][primary]
            data['quantization'] = {'primary': primary, 'rows': [
                {'build': name, 'file_mib': build['file_mb'],
                 'unjudged_at_10': quality['systems'][f'gguf_{name}']['unjudged@10'],
                 'condensed': quality['comparisons'][f'gguf_{name}']['nDCG@10_condensed'],
                 **quality['comparisons'][f'gguf_{name}']['nDCG@10']}
                for name, build in quant['builds'].items()]}
    if ann and ann.get('complete'):
        data['ann'] = {'queries': ann['queries'], 'sizes': ann['sizes'], 'synthetic': True}
        if ann_reference and ann_reference.get('complete'):
            if (ann_reference['source_sha256'] != ann['source_sha256']
                    or set(ann_reference['sizes']) != set(ann['sizes'])):
                raise ValueError('archive ANN audit does not match original vector inputs/sizes')
            data['ann']['archive_path'] = ann_reference['sizes']
    return data


def figure_names(data):
    return ['architecture', 'category_ndcg', 'judge_sensitivity'] + [name for key, name in
            (('latency', 'query_latency'), ('cost_quality', 'latency_quality'),
             ('quantization', 'quantization_quality'), ('ann', 'ann_scaling')) if key in data]


def _save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None,
                                                    'Title': name, 'Creator': 'paper.report.figures'})
    fig.savefig(directory / f'{name}.png', dpi=300, metadata={'Software': 'paper.report.figures'})


def render_figures(data, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import textwrap

    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10.5,
             'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
             'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}

    def note(fig, text, y=.025):
        lines = '\n'.join(textwrap.fill(line, width=84) for line in text.splitlines())
        fig.text(.02, y, lines, fontsize=10, va='bottom')

    with plt.rc_context(style):
        from matplotlib.patches import FancyBboxPatch
        fig, ax = plt.subplots(figsize=(160/25.4, 6.8))
        ax.set(xlim=(0, 10), ylim=(0, 11))
        ax.axis('off')

        def box(x, y, label, color='#EAF2F8', height=1.0):
            ax.add_patch(FancyBboxPatch((x-2.15, y-height/2), 4.3, height,
                                       boxstyle='round,pad=0.05', edgecolor='#30536B',
                                       facecolor=color, linewidth=1))
            ax.text(x, y, label, ha='center', va='center', fontsize=10)

        def arrow(start, end):
            ax.annotate('', xy=end, xytext=start,
                        arrowprops={'arrowstyle': '->', 'color': '#30536B', 'lw': 1.1})

        ax.text(2.3, 10.5, 'Incremental indexing', ha='center', weight='bold')
        ax.text(7.7, 10.5, 'Search', ha='center', weight='bold')
        for y, label in ((9.4, 'Archive files'), (7.9, 'Path / name exclusion filters'),
                         (6.4, 'Parse → redact text → chunk\nEmbed / reuse chunk vectors'),
                         (4.4, 'SQLite database\nDocuments · chunks · FTS5\nVectors for nonnumeric chunks')):
            box(2.3, y, label, height=1.5 if y == 4.4 else 1)
        for top, bottom in ((8.9, 8.4), (7.4, 6.9), (5.9, 5.2)):
            arrow((2.3, top), (2.3, bottom))
        ax.text(2.3, 2.8, 'File hashes skip unchanged files;\nchunk hashes reuse embeddings.\nNumeric-only chunks stay lexical.',
                ha='center', va='center', linespacing=1.5)
        for y, label in ((9.4, 'Query'), (7.9, 'Lexical normalization /\nretrieval embedding prompt'),
                         (6.4, 'FTS5 BM25 /\nexact vector similarity'), (4.4, 'Adaptive rule / RRF'),
                         (2.7, 'Ranked chunks → interface'), (1.1, 'Document collapse\n(benchmark only)')):
            box(7.7, y, label, color='#EAF5EF')
        for top, bottom in ((8.9, 8.4), (7.4, 6.9), (5.9, 4.9), (3.9, 3.2), (2.2, 1.6)):
            arrow((7.7, top), (7.7, bottom))
        ax.annotate('', xy=(5.5, 6.4), xytext=(4.5, 4.8),
                    arrowprops={'arrowstyle': '->', 'color': '#30536B', 'lw': 1.1,
                                'connectionstyle': 'arc3,rad=-.2'})
        fig.tight_layout()
        _save(fig, out, 'architecture')
        plt.close(fig)

        category = data['category']
        fig, ax = plt.subplots(figsize=(160/25.4, 4.3))
        x = np.arange(len(category['categories']))
        width = .18
        for i, system in enumerate(category['systems']):
            rows = [next(r for r in category['rows'] if r['category'] == c and r['system'] == system)
                    for c in category['categories']]
            means = np.array([r['mean'] for r in rows])
            errors = np.array([[r['mean'] - r['ci_low'] for r in rows], [r['ci_high'] - r['mean'] for r in rows]])
            ax.bar(x+(i-(len(category['systems'])-1)/2)*width, means, width, yerr=errors,
                   color=COLORS[i], label=system, hatch=('', '///', 'xx', '..')[i], edgecolor='white', linewidth=.4,
                   error_kw={'ecolor': '#303030', 'capsize': 2, 'elinewidth': .8})
        ax.set_xticks(x, [LABELS['en'][c].replace(' ', '\n').replace('Cross-lingual', 'Cross-\nlingual') +
                         f"\n(n={category['counts'][c]})"
                         for c in category['categories']])
        ax.set_ylim(0, 1)
        ax.set_ylabel('nDCG@10')
        ax.set_title(f"Category retrieval (n={category['queries']})\n{VARIANT_LABELS[category['primary']]}", pad=58)
        ax.legend(ncols=2, loc='lower center', bbox_to_anchor=(.5, 1.01), frameon=False)
        ax.yaxis.grid(True, alpha=.2, linewidth=.5)
        ax.set_axisbelow(True)
        note(fig, '95% query-bootstrap CIs. No-answer queries excluded.\nDepth-10 union pool; incomplete judgments, not full gold.')
        fig.tight_layout(rect=(0, .12, 1, 1))
        _save(fig, out, 'category_ndcg')
        plt.close(fig)

        sensitivity = data['sensitivity']
        fig, ax = plt.subplots(figsize=(160/25.4, 5.8))
        y = np.arange(len(sensitivity['systems']))
        for i, variant in enumerate(sensitivity['variants']):
            rows = [next(r for r in sensitivity['rows'] if r['variant'] == variant and r['system'] == s)
                    for s in sensitivity['systems']]
            errors = np.array([[r['mean'] - r['ci_low'] for r in rows], [r['ci_high'] - r['mean'] for r in rows]])
            offset = (i-(len(sensitivity['variants'])-1)/2)*.22
            ax.errorbar([r['mean'] for r in rows], y+offset, xerr=errors, fmt=('o','s','D')[i],
                        color=COLORS[i], markersize=4.2, capsize=2, elinewidth=.9, label=VARIANT_LABELS[variant])
        ax.set_yticks(y, sensitivity['systems'])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel('nDCG@10')
        ax.set_title(f"Judge sensitivity (n={sensitivity['queries']} answerable queries)", pad=85)
        ax.legend(loc='lower left', bbox_to_anchor=(-.35, 1.02), frameon=False, ncols=1)
        ax.xaxis.grid(True, alpha=.2, linewidth=.5)
        ax.set_axisbelow(True)
        status = 'Includes within-model Gemini review.' if 'gemini_reviewed' in sensitivity['variants'] else 'Initial judgments only; review not yet available.'
        note(fig, status + '\n95% query-bootstrap CIs; not independent gold. R@100 is pool-relative.')
        fig.tight_layout(rect=(0, .12, 1, 1))
        _save(fig, out, 'judge_sensitivity')
        plt.close(fig)

        if 'latency' in data:
            latency = data['latency']
            fig, ax = plt.subplots(figsize=(160/25.4, 4.7))
            y = np.arange(len(latency['systems']))
            for i, device in enumerate(('gpu', 'cpu')):
                rows = [next(r for r in latency['rows'] if r['device'] == device and r['system'] == s)
                        for s in latency['systems']]
                if any(not 0 < r['p50'] <= r['p95'] or r['n'] != 200 for r in rows):
                    raise ValueError('invalid latency measurement population or percentiles')
                pos = y + (i-.5)*.25
                ax.hlines(pos, [r['p50'] for r in rows], [r['p95'] for r in rows], color=COLORS[i], linewidth=1.3)
                ax.scatter([r['p50'] for r in rows], pos, color=COLORS[i], marker=('o', 's')[i], s=23,
                           label=f'{device.upper()} p50')
                ax.scatter([r['p95'] for r in rows], pos, facecolors='none', edgecolors=COLORS[i], s=20)
            ax.set_yticks(y, latency['systems'])
            ax.invert_yaxis()
            ax.set_xscale('log')
            ax.set_xlim(10, 450)
            ax.set_xticks([10, 20, 50, 100, 200, 400], ['10', '20', '50', '100', '200', '400'])
            ax.set_xlabel('Query latency (ms; logarithmic scale)')
            ax.set_title('Warm query latency on one laptop', pad=35)
            ax.grid(axis='x', alpha=.2)
            ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncols=2, frameon=False)
            note(fig, '200 warm-up + 200 timed queries per system/device.\nSegments show p50–p95, not confidence intervals. Desktop load was not controlled.')
            fig.tight_layout(rect=(0, .12, 1, 1))
            _save(fig, out, 'query_latency')
            plt.close(fig)

        if 'cost_quality' in data:
            quality = data['cost_quality']
            fig, ax = plt.subplots(figsize=(160/25.4, 5.5))
            offsets = ((8, -10), (8, -10), (8, 16), (-20, -16), (8, 14), (-22, 10), (8, -22))
            for i, row in enumerate(quality['rows']):
                ax.errorbar(row['gpu_p50_ms'], row['mean'],
                            yerr=[[row['mean']-row['ci_low']], [row['ci_high']-row['mean']]],
                            fmt='o', color=COLORS[i % len(COLORS)], capsize=2,
                            label=f"{i+1}: {row['system']}")
                ax.annotate(str(i+1), (row['gpu_p50_ms'], row['mean']), xytext=offsets[i],
                            textcoords='offset points', fontsize=10,
                            arrowprops={'arrowstyle': '-', 'color': '#555555', 'lw': .6})
            ax.set_xscale('log')
            ax.set_xlim(10, 220)
            ax.set_xticks([10, 20, 50, 100, 200], ['10', '20', '50', '100', '200'])
            ax.set_ylim(0, 1)
            ax.set_xlabel('Warm GPU query p50 (ms; logarithmic scale)')
            ax.set_ylabel('nDCG@10 (170 answerable queries)')
            ax.set_title('Frozen-run quality and current query cost', pad=85)
            ax.legend(ncols=2, loc='lower center', bbox_to_anchor=(.5, 1.01), frameon=False)
            ax.grid(alpha=.2)
            note(fig, 'Quality: frozen runs, reviewed Gemini labels, 95% query-bootstrap CIs.\nLatency: current implementation after the coverage fix; 200 timed queries.\nEmbedding and chunk search only. This is not a paired hardware experiment.')
            fig.tight_layout(rect=(0, .16, 1, 1))
            _save(fig, out, 'latency_quality')
            plt.close(fig)

        if 'quantization' in data:
            quant = data['quantization']
            fig, axes = plt.subplots(2, 1, figsize=(160/25.4, 5.8), sharex=True, sharey=True)
            for ax, condensed_run in zip(axes, (False, True)):
                for i, row in enumerate(quant['rows']):
                    stat = row['condensed'] if condensed_run else row
                    ax.errorbar(row['file_mib'], stat['mean'],
                                yerr=[[stat['mean']-stat['low']], [stat['high']-stat['mean']]],
                                fmt=('o', 's', 'D')[i], capsize=3, color=COLORS[i], label=row['build'])
                ax.axhline(0, color='#555555', linewidth=.8, linestyle='--')
                ax.set_ylim(-.035, .02)
                ax.set_ylabel('Δ nDCG@10 vs bf16')
                ax.set_title('Condensed runs: drop unjudged' if condensed_run else 'Unjudged treated as non-relevant')
                ax.grid(alpha=.2)
            axes[0].legend(frameon=False, ncols=3, loc='lower center', bbox_to_anchor=(.5, 1.18))
            axes[-1].set_xlabel('GGUF file size (MiB)')
            coverage = ', '.join(f"{r['build']} {100*r['unjudged_at_10']:.1f}%" for r in quant['rows'])
            note(fig, f"Reviewed Gemini labels; paired 95% query-bootstrap CIs.\nUnjudged@10: {coverage}.\nCondensing changes the evaluated ranking; neither panel is full gold.")
            fig.tight_layout(rect=(0, .15, 1, 1))
            _save(fig, out, 'quantization_quality')
            plt.close(fig)

        if 'ann' in data:
            ann = data['ann']
            sizes = sorted(map(int, ann['sizes']))
            rows = [ann['sizes'][str(n)] for n in sizes]
            fig, axes = plt.subplots(2, 1, figsize=(160/25.4, 5.8), sharex=True)
            for i, (key, label) in enumerate((('numpy_exact_ms', 'NumPy top-k'), ('faiss_flat_ms', 'Faiss flat'),
                                            ('hnsw_ef64', 'HNSW ef64'), ('hnsw_ef256', 'HNSW ef256'))):
                axes[0].plot(sizes, [r[key]['p50'] for r in rows], marker=('o','s','D','^')[i], color=COLORS[i], label=label)
            if 'archive_path' in ann:
                axes[0].plot(sizes, [ann['archive_path'][str(n)]['archive_numpy_exact_ms']['p50'] for n in sizes],
                             marker='v', color='#333333', linestyle='--', label='Archive vector path')
            for i, key in enumerate(('hnsw_ef64', 'hnsw_ef256')):
                axes[1].plot(sizes, [r[key]['recall@10'] for r in rows], marker=('D','^')[i], color=COLORS[i+2], label=key)
            axes[0].set_yscale('log')
            axes[0].set_ylabel('p50 single-query latency (ms)')
            axes[0].set_title('Synthetic vector scaling; not a real million-document archive')
            axes[0].legend(frameon=False, ncols=2)
            axes[1].set_ylabel('HNSW strict-ID recall@10')
            axes[1].set_ylim(0, 1.02)
            axes[1].set_xlabel('Vector count (logarithmic scale)')
            for ax in axes:
                ax.set_xscale('log')
                ax.grid(alpha=.2)
            note(fig, f"{ann['queries']} query vectors; noisy copies enlarge the real collection.\nEmbedding, lexical search and document collapse are excluded.\nNo uncertainty intervals estimated.", y=.02)
            fig.tight_layout(rect=(0, .12, 1, 1))
            _save(fig, out, 'ann_scaling')
            plt.close(fig)
