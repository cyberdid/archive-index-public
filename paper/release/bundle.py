"""Prepare a private, portable research candidate; never publish or grant a license.

Corpus bytes, query text, runs and grades remain unchanged. Only metadata paths
and exact references to changed hashes are normalized, with an explicit map.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from paper.reproduce import digest, input_files

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = 'paper/data/judgments/snapshot-2026-10-22'
REPLAY_REPORTS = ('ua_par.json', 'ua_par_gemini_initial.json', 'ua_par_gemma12_initial.json',
                  'ua_par_gemini_reviewed.json', 'judge_sensitivity.json', 'judges.json',
                  'encoder_control.json', 'covers_fix.json', 'efficiency.json', 'extension.json')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def copy(source, target):
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'not a regular input: {source.name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(source) != digest(target):
        raise ValueError('copy changed: ' + target.name)


def replace_hashes(value, replacements):
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [replace_hashes(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: replace_hashes(v, replacements) for k, v in value.items()}
    return value


def numbers(value, prefix=''):
    if isinstance(value, dict):
        return {p: n for k, v in value.items() for p, n in numbers(v, prefix + '/' + k).items()}
    if isinstance(value, list):
        return {p: n for i, v in enumerate(value) for p, n in numbers(v, prefix + '/' + str(i)).items()}
    return {prefix: value} if type(value) in (int, float, bool) else {}


def normalize(out, originals):
    for name in ('paper/data/ua-par/manifest.jsonl', SNAPSHOT + '/manifest.jsonl'):
        path = out / name
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            row['path'] = 'paper/data/ua-par/' + row['doc_id']
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    lock = json.loads((out / SNAPSHOT / 'inputs.json').read_text())
    for entry in lock['files']:
        path = out / SNAPSHOT / entry['snapshot']
        entry.update(sha256=digest(path), bytes=path.stat().st_size)
    for name in (SNAPSHOT + '/inputs.json', 'paper/results/judging_inputs_initial.json'):
        write_json(out / name, lock)
    # Propagate exact SHA references through the metadata DAG. No fuzzy deletion
    # of provenance keys and no numeric payload changes are permitted.
    candidates = [out / name for name in originals if Path(name).suffix in ('.json', '.jsonl')
                  and not name.startswith('paper/data/ua-par/')]
    base_values = {}
    for path in candidates:
        if path.name == 'manifest.jsonl':
            continue
        base_values[path] = ([json.loads(s) for s in path.read_text().splitlines() if s.strip()]
                             if path.suffix == '.jsonl' else json.loads(path.read_text()))
    substitutions = {}
    for _ in range(20):
        updated = {}
        for name, old in originals.items():
            current = digest(out / name)
            if current != old['sha256']:
                if old['sha256'] in updated and updated[old['sha256']] != current:
                    raise ValueError('identical original bytes exported inconsistently')
                updated[old['sha256']] = current
        if updated == substitutions:
            break
        substitutions = updated
        for path, value in base_values.items():
            if path.suffix == '.jsonl':
                normalized = replace_hashes(value, substitutions)
                current = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
                if normalized != current:
                    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in normalized))
            else:
                normalized = replace_hashes(value, substitutions)
                if normalized != json.loads(path.read_text()):
                    write_json(path, normalized)
    else:
        raise ValueError('metadata hash normalization did not converge')
    verified = []
    for name in REPLAY_REPORTS:
        original = json.loads((ROOT / 'paper/results' / name).read_text())
        exported = json.loads((out / 'paper/results' / name).read_text())
        if replace_hashes(original, substitutions) != exported or numbers(original) != numbers(exported):
            raise ValueError('export changed a report beyond exact provenance hashes: ' + name)
        verified.append(name)
    changed = {name: {'original_sha256': old['sha256'], 'export_sha256': digest(out / name),
                      'original_bytes': old['bytes'], 'export_bytes': (out / name).stat().st_size}
               for name, old in originals.items() if digest(out / name) != old['sha256']}
    write_json(out / 'NORMALIZATION_MAP.json', {
        'schema': 1, 'rule': 'Corpus path fields become paper/data/ua-par/<doc_id>; snapshot locks update '
                           'SHA/byte counts; exact old SHA references propagate through JSON metadata.',
        'changed_files': changed, 'exact_hash_substitutions': substitutions,
        'report_payloads_verified': verified, 'all_report_numeric_leaves_identical': True,
        'corpus_query_run_grade_bytes_changed': False,
        'scope': 'Metadata normalization changes snapshot provenance. Exported hashes are not original hashes.'})


def attributions(out, rows):
    raw = ROOT / 'paper/data/raw'
    target = out / 'paper/data/attributions'
    for row in rows:
        if row['part'] == 'tables':
            name = Path(row['doc_id']).stem + '.json'
            copy(raw / 'datagovua' / name, target / 'tables' / name)
    copy(raw / 'xm3600_attributions/image_attributions.csv', target / 'image_attributions.csv')
    copy(raw / 'xm3600_captions/captions.zip', target / 'captions.zip')
    repos = defaultdict(list)
    for row in rows:
        if row['part'] == 'code':
            repos[row['source']].append(row)
    inventory = []
    for url, files in sorted(repos.items()):
        notices = [r['doc_id'] for r in files
                   if any(s in Path(r['doc_id']).name.lower() for s in ('licen', 'copying', 'notice'))]
        repo = url.removeprefix('https://github.com/').replace('/', '__')
        copy(raw / 'github' / repo / 'meta.json', target / 'repositories' / (repo + '.json'))
        inventory.append({'repository': url, 'collected_files': len(files),
                          'repository_level_api_license': sorted({r['licence'] for r in files}),
                          'preserved_notices': notices})
    evidence = ROOT / 'paper/data/release-evidence'
    for path in sorted(evidence.glob('*')):
        copy(path, target / 'supplements' / path.name)
    write_json(target / 'inventory.json', {
        'repositories': inventory, 'tables': sum(r['part'] == 'tables' for r in rows),
        'images': sum(r['part'] == 'images' for r in rows),
        'per_file_attribution': 'paper/data/ua-par/manifest.jsonl; image_attributions.csv; tables/*.json',
        'license_scope': 'Repository API labels do not override embedded licenses or third-party notices.',
        'asset_license_overrides': {
            'code-uatranslit/app/src/main/res/font/rubik.ttf': {
                'license': 'SIL-OFL-1.1', 'evidence': 'supplements/Rubik.json',
                'license_text': 'supplements/Rubik-OFL.txt'}},
        'review_gates': [
            'Project-wide software license awaits author selection.',
            'transliteration-go lacked a collected license file; a pinned supplement matches all four frozen files.',
            'tts_uk preserves three dependency license files and its README MIT statement; no separate author LICENSE was collected.',
            'uaTranslit includes both MIT LICENSE and application TERMS.md; review scope before redistribution.',
        ], 'redistribution_approved': False})


def prepare(out):
    if out.exists():
        raise FileExistsError('choose a new candidate directory; never overwrite a prepared candidate')
    out.mkdir(parents=True)
    names = set(input_files('full'))
    tracked = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z',
                              '--', 'src', 'paper', 'tests'], cwd=ROOT, capture_output=True, check=True).stdout
    names.update(n for n in tracked.decode().split('\0') if n and not n.startswith('paper/data/'))
    names.update(('README.md', 'NOTICE', 'CITATION.cff', 'CHANGELOG.md', 'pyproject.toml', '.gitignore'))
    # Public snippets and blank grades only; this is not a completed human audit.
    import csv
    human = ROOT / 'paper/data/judgments/human-audit.tsv'
    with human.open(newline='') as stream:
        human_rows = list(csv.DictReader(stream, delimiter='\t'))
    if len(human_rows) != 150 or any(row['grade'] for row in human_rows):
        raise ValueError('expected the unchanged, ungraded 150-pair human audit sheet')
    names.add(human.relative_to(ROOT).as_posix())
    rows = [json.loads(s) for s in (ROOT / 'paper/data/ua-par/manifest.jsonl').read_text().splitlines()]
    seen = set()
    for row in rows:
        relative = Path(row['doc_id'])
        source = ROOT / 'paper/data/ua-par' / relative
        if relative.is_absolute() or '..' in relative.parts or row['doc_id'] in seen:
            raise ValueError('duplicate/escaping corpus document ID')
        seen.add(row['doc_id'])
        if digest(source) != row['sha256']:
            raise ValueError('corpus content differs from frozen manifest: ' + row['doc_id'])
        names.add('paper/data/ua-par/' + relative.as_posix())
    for name in sorted(names):
        copy(ROOT / name, out / name)
    originals = {name: {'sha256': digest(ROOT / name), 'bytes': (ROOT / name).stat().st_size}
                 for name in sorted(names)}
    normalize(out, originals)
    attributions(out, rows)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip())
    files = {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out / 'EXPORT_MANIFEST.json', {
        'schema': 1, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': commit, 'source_tree_dirty_at_preparation': dirty,
        'source_commit_role': 'Base commit; prepared_file_sha256 identifies actual exported bytes.',
        'corpus_files': len(rows), 'corpus_file_bytes_unchanged': True,
        'prepared_file_sha256': files, 'metadata_normalization': 'NORMALIZATION_MAP.json',
        'private_field_inputs': 'excluded; only aggregate JSON included', 'cloud_calls': 0,
        'release_approved': False, 'human_verified': False,
        'scope': 'Local review candidate. Prepared hashes precede replay/rendering; final SHA256SUMS bind sealed files.'})
    public_files = {}
    source_lines = {'paper/data/ua-par/' + row['doc_id']: index for index, row in enumerate(rows, 1)}
    for path in sorted(out.rglob('*')):
        if not path.is_file():
            continue
        name = path.relative_to(out).as_posix()
        if name in source_lines:
            basis = f'Frozen public corpus bytes verified against source manifest row {source_lines[name]}.'
        elif name.startswith('paper/data/attributions/'):
            basis = 'Public source attribution/caption metadata copied from the recorded source collection.'
        elif name.startswith('paper/data/'):
            basis = 'Frozen public query/run/grade/vector artifact; public document IDs/snippets derive from the corpus.'
        elif name == 'EXPORT_MANIFEST.json':
            basis = 'Prepared file inventory: paths and SHA-256 only; public corpus names come from its source manifest.'
        else:
            continue
        public_files[hashlib.sha256(name.encode()).hexdigest()] = {
            'sha256': digest(path), 'public_origin': True, 'basis': basis}
    from paper.release.check import REVIEWED_PUBLIC_MARKERS
    write_json(out / 'PUBLIC_COLLISION_POLICY.json', {
        'schema': 1, 'files': public_files,
        'reviewed_markers': REVIEWED_PUBLIC_MARKERS,
        'scope': 'Only four explicitly reviewed public-name/word collisions are eligible, identified in the scanner by SHA-256. '
                 'Every exception requires the exact public-origin file bytes. Full private queries and home paths '
                 'are always denied. Matching bytes remain in the data and every collision is reported.',
        'file_keys': 'SHA-256 of the complete relative path; corpus evidence refers to its manifest line.',
        'human_verified': False})
    print(f'Prepared {len(files)} files; {len(rows)} corpus files verified byte for byte. '
          'Next run full replay from this export, then seal.', flush=True)


def seal(out, run):
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest.get('status') != 'reproduction_complete' or len(manifest['stages']) != 14:
        raise ValueError('requires the completed 14-stage full replay')
    if manifest.get('source_export_sha256') != digest(out / 'EXPORT_MANIFEST.json'):
        raise ValueError('replay is not bound to this export')
    for name, expected in manifest['input_sha256'].items():
        if digest(run / 'inputs' / name) != expected:
            raise ValueError('replay input changed: ' + name)
    for stage in manifest['stages']:
        if stage['exit_code'] != 0 or digest(run / stage['log_file']) != stage['log_sha256']:
            raise ValueError('failed or changed stage: ' + stage['name'])
        for name, expected in stage['output_sha256'].items():
            if digest(run / name) != expected:
                raise ValueError('replay output changed: ' + name)
    for relative in ('paper/manuscript/generated',):
        shutil.copytree(run / 'work' / relative, out / relative, dirs_exist_ok=True)
    for name in ('article-en.pdf', 'article-uk.pdf', 'claims.csv', 'consistency_manifest.json'):
        copy(run / 'work/paper/manuscript' / name, out / 'paper/manuscript' / name)
    if not run.is_relative_to(out):
        shutil.copytree(run, out / 'reproduction', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '.execution.lock'))
    for folder in out.rglob('__pycache__'):
        shutil.rmtree(folder)
    from paper.release.check import ReleaseScanner, deny_list
    policy = json.loads((out / 'PUBLIC_COLLISION_POLICY.json').read_text())
    # Runtime inputs/work preserve the same public bytes at additional paths.
    if not run.is_relative_to(out):
        run = out / 'reproduction'
    for folder in ('inputs', 'work'):
        for name in manifest['input_sha256']:
            source_key = hashlib.sha256(name.encode()).hexdigest()
            if source_key in policy['files']:
                path = run / folder / name
                relative = path.relative_to(out).as_posix()
                policy['files'][hashlib.sha256(relative.encode()).hexdigest()] = {
                    **policy['files'][source_key], 'sha256': digest(path)}
    write_json(out / 'PUBLIC_COLLISION_POLICY_SEALED.json', policy)
    scanner = ReleaseScanner(deny_list(sorted((ROOT / 'evals').glob('queries*.jsonl')), str(Path.home())), policy)
    problems = scanner.scan(out)
    redacted = []
    for problem in problems:
        for marker in scanner.deny:
            problem = problem.replace(marker, '<private marker>')
            problem = problem.replace(' '.join(marker.split()), '<private marker>')
        redacted.append(problem)
    report = {'schema': 1, 'known_marker_count': len(scanner.deny), 'coverage': dict(scanner.counts),
              'problems': redacted, 'release_approved': False,
              'public_collisions': scanner.public_collisions,
              'scope': 'Known strings, filenames, decoded JSON, PDF text/metadata and supported compressed content. '
                       'No claim that marker matching proves absence of every possible private detail.'}
    write_json(out / 'PRIVACY_REPORT.json', report)
    if problems:
        for problem in redacted[:50]:
            print(problem, flush=True)
        raise ValueError(f'release refused: {len(problems)} inspection findings; see PRIVACY_REPORT.json')
    write_json(out / 'BUNDLE_MANIFEST.json', {
        'schema': 1, 'status': 'prepared_for_author_review',
        'export_manifest_sha256': digest(out / 'EXPORT_MANIFEST.json'),
        'normalization_map_sha256': digest(out / 'NORMALIZATION_MAP.json'),
        'privacy_report_sha256': digest(out / 'PRIVACY_REPORT.json'),
        'replay': json.loads((run / 'work/replay-verification.json').read_text()),
        'replay_manifest_sha256': digest(run / 'manifest.json'),
        'release_approved': False, 'human_verified': False,
        'pending': ['author license selection', 'third-party attribution/license review gates',
                    'human audit disposition',
                    'human source/declaration review', 'publication/DOI/submission approval']})
    files = sorted(p for p in out.rglob('*') if p.is_file() and p != out / 'SHA256SUMS')
    (out / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(out).as_posix()}\n' for p in files))
    print(f'Sealed local review candidate: {len(files)} checksums; zero unresolved findings; '
          f'{len(scanner.public_collisions)} hash-bound public-origin collisions recorded. Publication remains unapproved.')


def pack(out, archive):
    """Produce a deterministic local tarball, preserving file content and modes."""
    if archive.exists() or archive.is_relative_to(out):
        raise ValueError('archive must be a new file outside the candidate')
    if json.loads((out / 'BUNDLE_MANIFEST.json').read_text())['status'] != 'prepared_for_author_review':
        raise ValueError('seal the review candidate before packing')
    entries = {}
    for line in (out / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if digest(out / name) != expected:
            raise ValueError('sealed file changed: ' + name)
        entries[name] = expected
    actual = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    if actual != set(entries) | {'SHA256SUMS'}:
        raise ValueError('sealed file set changed')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + '.tmp')
    with temporary.open('xb') as stream, gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            for name in sorted(actual):
                path = out / name
                if path.is_symlink():
                    raise ValueError('refusing a symlink')
                info = tar.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                info.pax_headers = {}
                with path.open('rb') as source:
                    tar.addfile(info, source)
    temporary.rename(archive)
    archive.with_name(archive.name + '.sha256').write_text(f'{digest(archive)}  {archive.name}\n')
    policy_path = out / 'PUBLIC_COLLISION_POLICY_SEALED.json'
    if policy_path.is_file():
        policy = json.loads(policy_path.read_text())
        # The final inventory cannot hash itself in the sealed policy without
        # creating a cycle. Bind this derived inventory in an external sidecar
        # AFTER all its entries have been checked against actual sealed bytes.
        policy['files'][hashlib.sha256(b'SHA256SUMS').hexdigest()] = {
            'sha256': digest(out / 'SHA256SUMS'), 'public_origin': True,
            'basis': 'Final inventory: every path and checksum verified against the sealed public candidate.'}
        policy['archive_sha256'] = digest(archive)
        write_json(archive.with_name(archive.name + '.collision-policy.json'), policy)
    print(f'Created local archive, {archive.stat().st_size} bytes; checksum recorded. Not published.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'seal', 'pack'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT.is_relative_to(out):
        parser.error('candidate cannot contain the checkout')
    if args.command == 'prepare':
        prepare(out)
    elif args.command == 'pack':
        if args.archive is None:
            parser.error('pack requires --archive')
        pack(out, args.archive.resolve())
    elif args.run is None:
        parser.error('seal requires --run')
    else:
        seal(out, args.run.resolve())


if __name__ == '__main__':
    main()
