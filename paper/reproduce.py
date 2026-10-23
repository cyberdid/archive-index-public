"""Isolated, resumable quick rendering or full frozen-input statistical replay.

No model calls, private field inputs, index mutations, publication or tests are
implicit. Fresh physical measurement is documented separately in README_REPRODUCE.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
IMPORTED = ('corpus_summary.json', 'multimodal.json', 'field_aggregate.json', 'ann_scaling.json',
            'ann_reference_audit.json', 'quantization_reference.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, body):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def input_files(mode):
    files = {ROOT / 'pyproject.toml', ROOT / 'paper/run_all.sh', ROOT / 'paper/README_REPRODUCE.md'}
    for name in ('EXPORT_MANIFEST.json', 'NORMALIZATION_MAP.json', 'PUBLIC_COLLISION_POLICY.json'):
        if (ROOT / name).is_file():
            files.add(ROOT / name)
    files.update((ROOT / 'paper/environment').glob('*.txt'))
    files.add(ROOT / 'paper/manuscript/source_manifest.json')
    for folder in ('src', 'paper'):
        files.update(p for p in (ROOT / folder).rglob('*.py')
                     if 'data' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts)
    files.update(p for p in (ROOT / 'paper/manuscript/tex').rglob('*')
                 if p.is_file() and p.suffix in ('.tex', '.bib', '.sh'))
    files.update((ROOT / 'paper/results').glob('*.json'))
    if mode == 'full':
        data = ROOT / 'paper/data'
        for folder in ('judgments/snapshot-2026-10-22', 'runs', 'runs-control', 'runs-covers-fix',
                       'runs-gguf', 'runs-extension', 'judgments-extension'):
            files.update(p for p in (data / folder).rglob('*') if p.is_file())
        files.update(data / name for name in ('queries.tsv', 'queries-ext.tsv', 'queries-ext.log.jsonl',
                                              'ua-par/manifest.jsonl', 'ann/base.npy', 'ann/queries.npy'))
        files.update(data.glob('qrels*.txt'))
        files.update(data / 'judgments' / f'gemini-review{suffix}'
                     for suffix in ('.tsv', '.prepared.json', '.calls.jsonl'))
        files.update(data / 'judgments' / f'{judge}.tsv' for judge in ('gemini', 'gemma', 'gemma12', 'gpt'))
        for name in ('ann_scaling', 'ann_reference_audit'):
            source = json.loads((ROOT / f'paper/results/{name}.json').read_text())
            if not source.get('complete'):
                raise ValueError(f'full replay requires completed {name}')
        extension = json.loads((ROOT / 'paper/results/extension.json').read_text())
        if not extension.get('unique_wording_sensitivity'):
            raise ValueError('full replay requires the unique-wording extension sensitivity')
    missing = [p.relative_to(ROOT).as_posix() for p in files if not p.is_file()]
    if missing:
        raise ValueError('missing reproduction inputs: ' + ', '.join(sorted(missing)))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}


@contextmanager
def run_lock(directory):
    with (directory / '.execution.lock').open('w') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('this reproduction run is already active') from exc
        yield


def prepare(directory, mode, resume):
    fingerprint = input_files(mode)
    manifest_path = directory / 'manifest.json'
    interpreter = {'python': platform.python_version(), 'executable_sha256': digest(sys.executable),
                   'platform': platform.platform(),
                   'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()
                                           if d.metadata.get('Name')))}
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest['mode'] != mode or manifest['input_sha256'] != fingerprint or manifest['interpreter'] != interpreter:
            raise ValueError('resume inputs/code/environment changed; create a new isolated run')
        for name, expected in fingerprint.items():
            if digest(directory / 'inputs' / name) != expected:
                raise ValueError(f'immutable input snapshot changed: {name}')
        return manifest
    if manifest_path.exists():
        raise FileExistsError('existing run; use --resume or choose another directory')
    for name in fingerprint:
        for folder in ('inputs', 'work'):
            target = directory / folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            if digest(target) != fingerprint[name]:
                raise ValueError(f'input changed during copy: {name}')
    export = ROOT / 'EXPORT_MANIFEST.json'
    if export.is_file():
        origin = json.loads(export.read_text())
        source = {'source_commit': origin['source_commit'], 'source_tree_dirty': None,
                  'source_export_sha256': digest(export),
                  'source_commit_role': 'Original checkout base commit; this portable export has normalized metadata. '
                                        'Export manifest and input_sha256 identify the actual executed bytes.'}
    else:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        source = {'source_commit': git,
                  'source_commit_role': 'Base Git commit; input_sha256 records every executed source byte, including local edits.',
                  'source_tree_dirty': bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                                           capture_output=True, text=True, check=True).stdout.strip())}
    manifest = {'schema': 1, 'mode': mode, 'created_at_utc': now(),
                **source,
                'input_sha256': fingerprint, 'interpreter': interpreter, 'stages': [], 'status': 'prepared',
                'scope': __doc__.strip(), 'imported_measurements': list(IMPORTED),
                'efficiency_imports': 'CPU/GPU/index/GGUF timing and memory; only quality statistics replayed',
                'release_ready': False, 'human_verified': False}
    write_json(manifest_path, manifest)
    return manifest


def stage(directory, manifest, name, arguments, outputs):
    work, path = directory / 'work', directory / 'manifest.json'
    existing = next((s for s in manifest['stages'] if s['name'] == name and s['exit_code'] == 0), None)
    if existing:
        if any(not (directory / p).is_file() or digest(directory / p) != sha for p, sha in existing['output_sha256'].items()):
            raise ValueError(f'completed stage output changed: {name}')
        if digest(directory / existing['log_file']) != existing['log_sha256']:
            raise ValueError(f'completed stage log changed: {name}')
        print(f'Resume verified: {name}', flush=True)
        return
    attempt = 1 + sum(s['name'] == name for s in manifest['stages'])
    log = directory / 'logs' / f'{name}-{attempt:02d}.log'
    log.parent.mkdir(exist_ok=True)
    env = {**os.environ, 'PYTHONPATH': str(work / 'src') + os.pathsep + str(work), 'PAPER_PYTHON': sys.executable}
    command = ([sys.executable, *arguments[1:]] if arguments[0] == '{python}' else arguments)
    print(f'Running: {name}', flush=True)
    record = {'name': name, 'argv': arguments, 'cwd': 'work', 'started_at_utc': now(),
              'log_file': log.relative_to(directory).as_posix(), 'exit_code': None, 'output_sha256': {}}
    manifest['stages'].append(record)
    manifest['status'] = 'running'
    write_json(path, manifest)
    with log.open('w') as stream:
        completed = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
    original_log_sha256 = digest(log)
    content = log.read_text()
    for original, portable in ((str(work), '{RUN_WORK}'), (str(ROOT), '{SOURCE_ROOT}'),
                               (str(Path.home()), '{LOCAL_HOME}')):
        content = content.replace(original, portable)
    log.write_text(content)
    record.update(exit_code=completed.returncode, finished_at_utc=now(), log_sha256=digest(log),
                  log_original_sha256=original_log_sha256,
                  log_normalization='Run/source/home paths replaced with portable placeholders before hashing.')
    if completed.returncode:
        manifest['status'] = 'failed'
        write_json(path, manifest)
        raise RuntimeError(f'{name} exited {completed.returncode}; inspect {log}')
    selected = [p for pattern in outputs for p in work.glob(pattern) if p.is_file()]
    if not selected:
        selected = [log]
    record['output_sha256'] = {p.relative_to(directory).as_posix(): digest(p) for p in selected}
    write_json(path, manifest)


def execute(directory, manifest):
    work = directory / 'work'
    py = '{python}'
    if manifest['mode'] == 'full':
        steps = [
            ('snapshot', [py, '-m', 'paper.judge.finalize', 'validate'], []),
            ('qrels', [py, '-m', 'paper.judge.finalize', 'merge'], ['paper/data/qrels*.txt', 'paper/data/provenance.json']),
            ('sensitivity', [py, '-m', 'paper.eval.judge_sensitivity', '--review', 'paper/data/judgments/gemini-review.tsv'],
             ['paper/results/ua_par_*.json', 'paper/results/judge_sensitivity.*']),
            ('primary', [py, '-m', 'paper.replay', 'primary'], ['paper/results/ua_par.json']),
            ('judges', [py, '-m', 'paper.judge.report'], ['paper/results/judges.json']),
            ('controls', [py, '-m', 'paper.replay', 'controls'], ['paper/results/encoder_control.json']),
            ('coverage', [py, '-m', 'paper.replay', 'covers'], ['paper/results/covers_fix.json']),
            ('gguf-quality', [py, '-m', 'paper.replay', 'quantization'], ['paper/results/efficiency.json']),
            ('extension', [py, '-m', 'paper.eval.extension', 'evaluate'], ['paper/results/extension.json']),
            ('replay-equality', [py, '-m', 'paper.replay', 'verify'], ['replay-verification.json']),
        ]
        for name, args, outputs in steps:
            stage(directory, manifest, name, args, outputs)
    stage(directory, manifest, 'generate', [py, '-m', 'paper.report.generate'], ['paper/manuscript/generated/**/*'])
    stage(directory, manifest, 'validate', [py, '-m', 'paper.report.validate'], [])
    stage(directory, manifest, 'pdf', ['bash', 'paper/manuscript/tex/build.sh', 'all'], ['paper/manuscript/article-*.pdf'])
    stage(directory, manifest, 'claims', [py, '-m', 'paper.report.claim_audit'],
          ['paper/manuscript/claims.csv', 'paper/manuscript/consistency_manifest.json'])
    # Report the actual release gate separately from successful reproduction.
    # Rebuilding a still-unapproved draft is allowed; it must not become a release certificate.
    from paper.release.compliance import check_pdf
    manifest['pdf_compliance'] = {lang: check_pdf(directory / f'work/paper/manuscript/article-{lang}.pdf',
                                     directory / f'work/paper/manuscript/tex/{lang}/main.log') for lang in ('en', 'uk')}
    manifest.update(status='reproduction_complete', finished_at_utc=now(), release_ready=False)
    # Compiler diagnostics and Python bytecode can contain absolute paths. They
    # are derived runtime files, not frozen scientific inputs.
    normalized_logs = dict(manifest.get('runtime_export', {}).get('normalized_logs', {}))
    for log in work.rglob('*.log'):
        original_sha = digest(log)
        content = log.read_text()
        for original, portable in ((str(work), '{RUN_WORK}'), (str(ROOT), '{SOURCE_ROOT}'),
                                   (str(Path.home()), '{LOCAL_HOME}')):
            content = content.replace(original, portable)
        if content != log.read_text():
            log.write_text(content)
            normalized_logs[log.relative_to(directory).as_posix()] = {
                'original_sha256': original_sha, 'export_sha256': digest(log)}
    for folder in work.rglob('__pycache__'):
        shutil.rmtree(folder)
    manifest['runtime_export'] = {'normalized_logs': normalized_logs,
                                 'bytecode': 'Derived __pycache__ directories excluded.'}
    write_json(directory / 'manifest.json', manifest)
    artifacts = sorted(p for p in directory.rglob('*') if p.is_file()
                       and p.name not in ('SHA256SUMS', '.execution.lock') and p.suffix != '.tmp')
    (directory / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(directory).as_posix()}\n'
                                               for p in artifacts))
    print(f'Reproduction complete: {directory}\nRelease approval remains separate; see manifest.json.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('quick', 'full'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and args.out is None:
        parser.error('--resume requires --out')
    directory = (args.out or ROOT / 'paper/data/reproduction' /
                 (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])).resolve()
    if ROOT == directory or ROOT.is_relative_to(directory):
        parser.error('run directory cannot contain the source checkout')
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        manifest = prepare(directory, args.mode, args.resume)
        execute(directory, manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
