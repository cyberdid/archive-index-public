"""Recompute public statistical analyses from frozen runs/grades, without models.

Physical resource, multimodal and private-field measurements remain explicitly
imported artifacts. Replaying statistics is not a claim to have remeasured them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def quality_inputs():
    from paper.eval.controls import QRELS_VARIANTS
    from paper.eval.metrics import read_qrels, read_run
    from paper.queries.schema import load_queries
    queries = load_queries(Path('paper/data/queries.tsv'))
    answerable = {q.qid for q in queries if q.category != 'no_answer'}
    labels = {variant: {qid: judged for qid, judged in read_qrels(Path('paper/data') / filename).items()
                        if qid in answerable}
              for variant, filename in QRELS_VARIANTS.items() if (Path('paper/data') / filename).is_file()}
    runs = {path.stem: read_run(path) for path in Path('paper/data/runs').glob('*.trec')}
    return queries, labels, runs


def controls():
    from paper.eval.controls import COMPARED, compare_encoders
    from paper.eval.metrics import read_run
    _, labels, runs = quality_inputs()
    runs.update({p.stem: read_run(p) for p in Path('paper/data/runs-control').glob('*.trec')})
    selected = {name: runs[name] for name in COMPARED}
    original = read_json('paper/results/encoder_control.json')
    original['variants'] = {variant: compare_encoders(qrels, selected) for variant, qrels in labels.items()}
    write_json('paper/results/encoder_control.json', original)


def covers():
    from paper.eval.controls import compare_encoders
    from paper.eval.metrics import read_run
    queries, labels, runs = quality_inputs()
    names = [name for name in runs if name.startswith('adaptive')]
    selected = {name: runs[name] for name in ('dense_eg2', 'dense_e5', *names)}
    fixed = {p.stem + '_fixed': read_run(p) for p in Path('paper/data/runs-covers-fix').glob('*.trec')}
    selected.update(fixed)
    def top(run, qid):
        return sorted(run.get(qid, {}), key=lambda doc: -run[qid][doc])[:10]
    original = read_json('paper/results/covers_fix.json')
    original['queries_with_changed_top10'] = {
        name: sum(top(runs[name], q.qid) != top(selected[name + '_fixed'], q.qid) for q in queries)
        for name in names}
    original['variants'] = {variant: compare_encoders(qrels, selected, reference='adaptive_fixed')
                            for variant, qrels in labels.items()}
    write_json('paper/results/covers_fix.json', original)


def quantization():
    from paper.eval.controls import compare_encoders
    from paper.eval.metrics import read_run
    _, labels, _ = quality_inputs()
    runs = {'dense_eg2_text': read_run(Path('paper/data/runs-control/dense_eg2_text.trec'))}
    runs.update({'gguf_' + p.stem: read_run(p) for p in Path('paper/data/runs-gguf').glob('*.trec')})
    original = read_json('paper/results/efficiency.json')
    quant = original['quantization']['retrieval']
    quant['quality'] = {variant: compare_encoders(qrels, runs, reference='dense_eg2_text')
                        for variant, qrels in labels.items()}
    write_json('paper/results/efficiency.json', original)


def primary():
    shutil.copyfile('paper/results/ua_par_gemini_reviewed.json', 'paper/results/ua_par.json')


def verify():
    import hashlib
    baseline = Path('../inputs/paper/results')
    names = ('ua_par.json', 'ua_par_gemini_initial.json', 'ua_par_gemma12_initial.json',
             'ua_par_gemini_reviewed.json', 'judge_sensitivity.json', 'judges.json',
             'encoder_control.json', 'covers_fix.json', 'efficiency.json', 'extension.json')
    for name in names:
        if read_json(baseline / name) != read_json(Path('paper/results') / name):
            raise ValueError(f'recomputed report differs from the archived report: {name}')
    for filename in ('base.npy', 'queries.npy'):
        with (Path('paper/data/ann') / filename).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        expected = read_json('paper/results/ann_scaling.json')['source_sha256'][filename]
        if actual != expected:
            raise ValueError(f'ANN vector input changed: {filename}')
    if Path('paper/data/qrels.txt').read_bytes() != Path('../inputs/paper/data/qrels.txt').read_bytes():
        raise ValueError('replayed canonical qrels differ from archived canonical qrels')
    Path('replay-verification.json').write_text(json.dumps({
        'equal_reports': list(names), 'canonical_qrels_equal': True, 'ann_vector_hashes_equal': True,
        'physical_measurements': 'imported, not remeasured', 'cloud_calls': 0}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('primary', 'controls', 'covers', 'quantization', 'verify'))
    args = parser.parse_args()
    globals()[args.command]()
    print(f'Finished frozen-input analysis: {args.command}')
