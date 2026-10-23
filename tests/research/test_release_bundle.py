import hashlib
import json
import tarfile

import pytest

from paper.release.bundle import pack
from paper.release.check import ReleaseScanner, REVIEWED_PUBLIC_MARKERS, check_release


def _sealed(tmp_path):
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    (candidate / 'BUNDLE_MANIFEST.json').write_text(json.dumps({'status': 'prepared_for_author_review'}))
    (candidate / 'hello.txt').write_text('Frozen public content.\n')
    (candidate / 'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
        for p in sorted(candidate.iterdir())))
    return candidate


def test_archive_is_deterministic_and_its_contents_are_scannable(tmp_path):
    candidate = _sealed(tmp_path)
    archives = tmp_path / 'archives'
    first, second = archives / 'first.tar.gz', archives / 'second.tar.gz'
    pack(candidate, first)
    pack(candidate, second)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as tar:
        assert tar.extractfile('hello.txt').read() == b'Frozen public content.\n'
        assert all(m.uid == m.gid == m.mtime == 0 for m in tar)
    assert check_release(archives, ['/home/example-person']) == []


def test_pack_refuses_changed_bytes_or_extra_files(tmp_path):
    candidate = _sealed(tmp_path)
    (candidate / 'hello.txt').write_text('changed')
    with pytest.raises(ValueError, match='sealed file changed'):
        pack(candidate, tmp_path / 'changed.tar.gz')


def test_pack_refuses_an_unrecorded_file(tmp_path):
    candidate = _sealed(tmp_path)
    (candidate / 'extra.txt').write_text('extra')
    with pytest.raises(ValueError, match='sealed file set changed'):
        pack(candidate, tmp_path / 'extra.tar.gz')


def test_archive_inventory_has_external_hash_bound_collision_evidence(tmp_path):
    candidate = _sealed(tmp_path)
    name = bytes([65, 112, 112, 108, 101]).decode() + '.txt'
    (candidate / name).write_text('Public synthetic source.')
    public_files = {hashlib.sha256(name.encode()).hexdigest(): {
        'sha256': hashlib.sha256((candidate / name).read_bytes()).hexdigest(),
        'public_origin': True, 'basis': 'Synthetic public source.'}}
    (candidate / 'PUBLIC_COLLISION_POLICY_SEALED.json').write_text(json.dumps({
        'files': public_files, 'reviewed_markers': REVIEWED_PUBLIC_MARKERS}))
    (candidate / 'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
        for p in sorted(candidate.iterdir()) if p.name != 'SHA256SUMS'))
    archive = tmp_path / 'archives' / 'review.tar.gz'
    pack(candidate, archive)
    policy = json.loads(archive.with_name(archive.name + '.collision-policy.json').read_text())
    assert ReleaseScanner([name[:-4]], policy).scan(archive.parent) == []
