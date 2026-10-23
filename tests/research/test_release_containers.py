import gzip
import hashlib
import json
from types import SimpleNamespace
import zipfile

from paper.release.check import ReleaseScanner, check_release, literal_pattern


PRIVATE = '/home/example-person'
PUBLIC_WORD = bytes([65, 112, 112, 108, 101]).decode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _policy(name, data):
    return {'reviewed_markers': {h: 'Reviewed public-source collision.' for h in (
                _sha(PUBLIC_WORD.encode()),
                '6ca5cab77e702c787b4c14b3d3bf26bad43da606be6eed04ab0b9720120ae081',
                '85fb49e051915a77ef6d27b52ce52eb5c4f81d46e18e1526eb59ddb150289ea7',
                '1ecd971c816da4de083c08d9aa816cdbb9511e383df9c94681f61ddfa7f4612a')},
            'files': {_sha(name.encode()): {'sha256': _sha(data), 'public_origin': True,
                                          'basis': 'Synthetic public-origin evidence for the unit fixture.'}}}


def test_nested_zip_gzip_and_escaped_json_are_inspected(tmp_path):
    payload = json.dumps({'path': PRIVATE}, ensure_ascii=True).replace('/', '\\u002f').encode()
    with zipfile.ZipFile(tmp_path / 'outer.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('inner.json.gz', gzip.compress(payload))
    scanner = ReleaseScanner([PRIVATE])
    assert scanner.scan(tmp_path)
    assert scanner.counts['zip_containers'] == 1 and scanner.counts['gzip_containers'] == 1


def test_utf16_ascii_marker_cannot_hide_in_utf8_decodable_bytes(tmp_path):
    (tmp_path / 'utf16.txt').write_bytes(PRIVATE.encode('utf-16-le'))
    assert check_release(tmp_path, [PRIVATE])


def test_filename_and_symlink_are_inspected(tmp_path):
    (tmp_path / 'sensitive-project.txt').write_text('ordinary text')
    (tmp_path / 'link').symlink_to('sensitive-project.txt')
    found = check_release(tmp_path, ['sensitive-project'])
    assert any('filename' in p for p in found) and any('symlink refused' in p for p in found)


def test_bom_json_is_decoded_and_its_strings_inspected(tmp_path):
    value = json.dumps({'path': PRIVATE}).replace('/', '\\u002f')
    (tmp_path / 'bom.json').write_text('\ufeff' + value)
    found = check_release(tmp_path, [PRIVATE])
    assert any('decoded-json' in p for p in found)
    assert not any('unreadable JSON' in p for p in found)


def test_invalid_container_and_expansion_limit_fail_closed(tmp_path):
    (tmp_path / 'bad.zip').write_bytes(b'PK\x03\x04broken')
    (tmp_path / 'large.gz').write_bytes(gzip.compress(b'x' * 100))
    scanner = ReleaseScanner([PRIVATE])
    scanner.MAX_MEMBER = 20
    found = scanner.scan(tmp_path)
    assert any('bad.zip' in p and 'failed' in p for p in found)
    assert any('large.gz' in p and 'failed' in p for p in found)


def test_pdf_text_and_metadata_are_inspected(monkeypatch, tmp_path):
    seen = []

    def execute(command, **kwargs):
        seen.append(command)
        text = PRIVATE if command[0] == 'pdfinfo' else 'ordinary text'
        return SimpleNamespace(returncode=0, stdout=text.encode())

    monkeypatch.setattr('paper.release.check.subprocess.run', execute)
    (tmp_path / 'document.pdf').write_bytes(b'%PDF-synthetic')
    assert check_release(tmp_path, [PRIVATE])
    assert any('-meta' in c for c in seen) and any('-custom' in c for c in seen)


def test_public_collision_requires_exact_file_hash_and_is_reported(tmp_path):
    data = (PUBLIC_WORD + ' is a public company/fruit name.').encode()
    path = tmp_path / 'public.txt'
    path.write_bytes(data)
    assert check_release(tmp_path, [PUBLIC_WORD])  # No default exception.
    scanner = ReleaseScanner([PUBLIC_WORD], _policy('public.txt', data))
    assert scanner.scan(tmp_path) == [] and scanner.public_collisions
    path.write_bytes(data + b' changed')
    assert ReleaseScanner([PUBLIC_WORD], _policy('public.txt', data)).scan(tmp_path)


def test_public_collision_never_exempts_home_or_full_private_query(tmp_path):
    query = PUBLIC_WORD + ' private synthetic query text'
    data = (query + '\n' + PRIVATE).encode()
    (tmp_path / 'public.txt').write_bytes(data)
    scanner = ReleaseScanner([PUBLIC_WORD, query, PRIVATE], _policy('public.txt', data))
    found = scanner.scan(tmp_path)
    assert any(query in p for p in found) and any(PRIVATE in p for p in found)


def test_reviewed_short_public_names_require_explicit_marker_evidence(tmp_path):
    # Public recreation captions / Scandinavian-name prefixes; never actual queries.
    words = [bytes([103, 97, 109, 101]).decode(), bytes([78, 111, 114, 100]).decode(),
             bytes([69, 83, 69, 84]).decode()]
    data = (' / '.join(words)).encode()
    (tmp_path / 'public.txt').write_bytes(data)
    policy = _policy('public.txt', data)
    assert ReleaseScanner(words, policy).scan(tmp_path) == []
    policy['reviewed_markers'] = {}
    assert ReleaseScanner(words, policy).scan(tmp_path)


def test_arbitrary_marker_cannot_be_added_to_collision_policy(tmp_path):
    data = b'sensitive-project'
    (tmp_path / 'public.txt').write_bytes(data)
    policy = _policy('public.txt', data)
    policy['reviewed_markers'][_sha(data)] = 'Unreviewed arbitrary marker.'
    assert ReleaseScanner([data.decode()], policy).scan(tmp_path)


def test_collision_locations_redact_every_known_public_marker(tmp_path):
    other = bytes([103, 97, 109, 101]).decode()
    name = other + '-' + PUBLIC_WORD + '.txt'
    data = (other + ' ' + PUBLIC_WORD).encode()
    (tmp_path / name).write_bytes(data)
    scanner = ReleaseScanner([other, PUBLIC_WORD], _policy(name, data))
    assert scanner.scan(tmp_path) == []
    text = json.dumps(scanner.public_collisions)
    assert other not in text and PUBLIC_WORD not in text


def test_literal_pattern_handles_shared_prefixes_and_binary_needles():
    pattern = literal_pattern(['abc', 'abcdef', 'bcd'])
    assert pattern.search('abcdef').group() == 'abcdef'
    assert literal_pattern([b'\x00abc', b'\x00abd']).search(b'x\x00abd').group() == b'\x00abd'
    assert literal_pattern([]) is None
