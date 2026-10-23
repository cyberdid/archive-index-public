"""Release guard (plan Task 11): nothing private may leave in a release directory.

The deny list holds the home path, the names of private projects the private evaluation sets point at, and the
private query texts themselves. Check filenames, bytes, decoded JSON, extracted
PDF text/metadata and recursively expanded ZIP, tar, gzip and deflate content.
This is a known-marker guard, not a guarantee against all possible disclosures.
"""

from __future__ import annotations

import json
import gzip
import hashlib
import io
import re
import subprocess
import tarfile
import zipfile
import zlib
from collections import Counter
from pathlib import Path

MIN_MARKER = 4  # shorter strings would match ordinary text

# Explicitly investigated public-source collisions, identified without exposing
# private project names. Never infer new exceptions from a failed scan.
REVIEWED_PUBLIC_MARKERS = {
    'f223faa96f22916294922b171a2696d868fd1f9129302eb41a45b2a2ea2ebbfd':
        'Public company/fruit name in Wikipedia, captions and upstream templates.',
    '6ca5cab77e702c787b4c14b3d3bf26bad43da606be6eed04ab0b9720120ae081':
        'Common English word in public recreation captions/document IDs and contributor names.',
    '85fb49e051915a77ef6d27b52ce52eb5c4f81d46e18e1526eb59ddb150289ea7':
        'Four-letter prefix in public Scandinavian names and caption titles.',
    '1ecd971c816da4de083c08d9aa816cdbb9511e383df9c94681f61ddfa7f4612a':
        'Public antivirus vendor name and a substring of an uppercase reset command in upstream sources.',
}


def literal_pattern(needles):
    """Factor shared prefixes so a large marker list scans each blob once."""
    needles = set(needles)
    if not needles:
        return None
    empty = next(iter(needles))[:0]
    root = {}
    for needle in needles:
        node = root
        for index in range(len(needle)):
            node = node.setdefault(needle[index:index + 1], {})
        node[None] = {}

    def expression(node):
        prefix = empty
        while len(node) == 1 and None not in node:
            char, node = next(iter(node.items()))
            prefix += re.escape(char)
        branches = [re.escape(char) + expression(child) for char, child in node.items() if char is not None]
        if not branches:
            return prefix
        opening, separator, closing, optional = ((b'(?:', b'|', b')', b'?') if isinstance(empty, bytes)
                                                else ('(?:', '|', ')', '?'))
        body = opening + separator.join(branches) + closing
        return prefix + body + (optional if None in node else empty)

    return re.compile(expression(root))


def deny_list(private_query_files: list[Path], home: str) -> list[str]:
    deny = {home}
    for path in private_query_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            for key in ("query", "expect_phrase"):
                if len(q.get(key, "")) >= 12:
                    deny.add(q[key])
            if q.get("expect_project"):
                deny.add(q["expect_project"])
            match = re.search(r"/Projects/([^/]+)/", q.get("source_path", ""))
            if match:
                deny.add(match.group(1))
    return sorted(d for d in deny if len(d) >= MIN_MARKER)


class ReleaseScanner:
    """Fail closed on unreadable containers and bounded decompression limits."""

    MAX_MEMBER = 512 * 1024 * 1024
    MAX_EXPANDED = 8 * 1024 * 1024 * 1024
    MAX_DEPTH = 8

    def __init__(self, deny, public_collisions=None):
        self.deny = deny
        self.pattern = literal_pattern(deny)
        self.folded_pattern = literal_pattern(' '.join(d.split()) for d in deny)
        self.encoded = {d.encode(encoding): d for d in deny
                        for encoding in ('utf-8', 'utf-16-le', 'utf-16-be')}
        self.byte_pattern = literal_pattern(self.encoded)
        self.problems = []
        self.counts = Counter()
        self.public_policy = public_collisions or {}
        self.public_collisions = []
        self.current_file = None
        self.current_sha256 = None

    def hit(self, location, marker):
        marker_hash = hashlib.sha256(marker.encode()).hexdigest()
        path_hash = hashlib.sha256((self.current_file or '').encode()).hexdigest()
        approved = self.public_policy.get('files', {}).get(path_hash, {})
        # A fixed reviewed marker registry AND exact public-origin file bytes
        # are required. Home paths and private query strings have no exception.
        if (marker_hash in REVIEWED_PUBLIC_MARKERS
                and self.public_policy.get('reviewed_markers', {}).get(marker_hash)
                and approved.get('sha256') == self.current_sha256
                and approved.get('public_origin') and approved.get('basis')):
            public_location = location
            public_file = self.current_file
            for known_marker in self.deny:
                public_location = public_location.replace(known_marker, '<public name>')
                public_file = public_file.replace(known_marker, '<public name>')
            self.public_collisions.append({'location': public_location,
                                           'marker_sha256': marker_hash,
                                           'file': public_file,
                                           'file_path_sha256': path_hash, 'file_sha256': self.current_sha256,
                                           'basis': approved['basis']})
        else:
            self.problems.append(f'{location}: {marker}')

    def text(self, label, value, whitespace=False):
        if whitespace:
            value = ' '.join(value.split())
        pattern = self.folded_pattern if whitespace else self.pattern
        if pattern:
            for match in pattern.finditer(value):
                self.hit(f'{label}:{value.count(chr(10), 0, match.start()) + 1}', match.group())

    def strings(self, value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from self.strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from self.strings(child)

    def expanded(self, label, data, depth):
        self.counts['expanded_bytes'] += len(data)
        if (len(data) > self.MAX_MEMBER or self.counts['expanded_bytes'] > self.MAX_EXPANDED
                or depth > self.MAX_DEPTH):
            raise ValueError('archive expansion limit exceeded')
        self.blob(label, data, depth)

    def blob(self, label, data, depth=0):
        self.text(label + ':filename', label)
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            if self.byte_pattern:
                for match in self.byte_pattern.finditer(data):
                    self.hit(label, self.encoded[match.group()])
            self.counts['binary_blobs'] += 1
        else:
            self.text(label, text)
            if '\x00' in text and self.byte_pattern:
                for match in self.byte_pattern.finditer(data):
                    self.hit(label, self.encoded[match.group()])
            self.counts['text_blobs'] += 1
            if Path(label).suffix in ('.json', '.jsonl', '.ipynb'):
                try:
                    values = ([json.loads(line.lstrip('\ufeff')) for line in text.splitlines() if line.strip()]
                              if label.endswith('.jsonl') else [json.loads(text.lstrip('\ufeff'))])
                    for value in values:
                        for string in self.strings(value):
                            self.text(label + ':decoded-json', string)
                    self.counts['decoded_json'] += 1
                except (ValueError, TypeError):
                    self.problems.append(f'{label}: <unreadable JSON>')
        try:
            if data.startswith(b'%PDF-'):
                for command in (['pdftotext', '-enc', 'UTF-8', '-', '-'],
                                ['pdfinfo', '-meta', '-'], ['pdfinfo', '-custom', '-']):
                    result = subprocess.run(command, input=data, capture_output=True, timeout=90)
                    if result.returncode:
                        raise ValueError(f'{command[0]} failed')
                    self.text(label + ':' + command[0], result.stdout.decode('utf-8'), whitespace=True)
                self.counts['pdfs'] += 1
            if data.startswith((b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08')):
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for member in archive.infolist():
                        self.text(label + ':zip-filename', member.filename)
                        if not member.is_dir():
                            if member.file_size > self.MAX_MEMBER:
                                raise ValueError('ZIP member too large')
                            self.expanded(label + '!' + member.filename, archive.read(member), depth + 1)
                    self.counts['zip_containers'] += 1
            elif data.startswith(b'\x1f\x8b'):
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as archive:
                    unpacked = archive.read(self.MAX_MEMBER + 1)
                inner_name = Path(label).name.removesuffix('.gz')
                self.expanded(label + '!' + inner_name, unpacked, depth + 1)
                self.counts['gzip_containers'] += 1
            elif data[257:262] == b'ustar':
                with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
                    for member in archive:
                        self.text(label + ':tar-filename', member.name)
                        if member.issym() or member.islnk():
                            raise ValueError('archive contains a link')
                        if member.isfile():
                            if member.size > self.MAX_MEMBER:
                                raise ValueError('tar member too large')
                            self.expanded(label + '!' + member.name, archive.extractfile(member).read(), depth + 1)
                    self.counts['tar_containers'] += 1
            elif label.endswith('.deflate'):
                # The corpus dictionaries use a raw deflate stream.
                unpacker = zlib.decompressobj(-zlib.MAX_WBITS)
                unpacked = unpacker.decompress(data, self.MAX_MEMBER + 1)
                if not unpacker.eof or unpacker.unused_data:
                    raise ValueError('incomplete or excessive deflate stream')
                self.expanded(label + '!deflate', unpacked, depth + 1)
                self.counts['deflate_containers'] += 1
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, zipfile.BadZipFile,
                tarfile.TarError, zlib.error) as exc:
            self.problems.append(f'{label}: <container/PDF inspection failed: {type(exc).__name__}>')

    def scan(self, root):
        if not root.is_dir():
            raise ValueError('release root must be a directory')
        for path in sorted(root.rglob('*')):
            label = path.relative_to(root).as_posix()
            self.current_file = label
            self.current_sha256 = None
            if path.is_file() and not path.is_symlink():
                with path.open('rb') as stream:
                    self.current_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
            self.text(label + ':filename', label)
            if path.is_symlink():
                self.problems.append(f'{label}: <symlink refused>')
            elif path.is_file():
                self.counts['files'] += 1
                if label.endswith(('.tar.gz', '.tgz', '.tar')):
                    # A release tarball may exceed the per-member memory limit.
                    # Inspect its contents as a stream instead of buffering it.
                    try:
                        with tarfile.open(path, mode='r|*') as archive:
                            for member in archive:
                                self.current_file = member.name
                                self.current_sha256 = None
                                if member.issym() or member.islnk():
                                    raise ValueError('archive contains a link')
                                if member.isfile():
                                    if member.size > self.MAX_MEMBER:
                                        raise ValueError('tar member too large')
                                    data = archive.extractfile(member).read()
                                    self.current_sha256 = hashlib.sha256(data).hexdigest()
                                    self.text(label + ':tar-filename', member.name)
                                    self.expanded(label + '!' + member.name, data, 1)
                                elif not member.isdir():
                                    raise ValueError('archive contains a special file')
                            self.counts['tar_containers'] += 1
                    except (OSError, ValueError, tarfile.TarError) as exc:
                        self.problems.append(f'{label}: <tar inspection failed: {type(exc).__name__}>')
                else:
                    self.blob(label, path.read_bytes())
        return sorted(set(self.problems))


def check_release(root: Path, deny: list[str]) -> list[str]:
    """Return marker/error locations; the caller must redact markers before export."""
    return ReleaseScanner(deny).scan(root)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--public-collisions', type=Path,
                        help='explicit hash-bound public-origin collision policy; default is no exceptions')
    args = parser.parse_args()
    root = args.root
    files = sorted(Path("evals").glob("queries*.jsonl"))
    deny = deny_list(files, str(Path.home()))
    policy = json.loads(args.public_collisions.read_text()) if args.public_collisions else None
    scanner = ReleaseScanner(deny, policy)
    problems = scanner.scan(root)
    redacted = []
    for problem in problems:
        for marker in deny:
            problem = problem.replace(marker, '<private marker>')
            problem = problem.replace(' '.join(marker.split()), '<private marker>')
        redacted.append(problem)
        if len(redacted) <= 50:
            print(problem)
    if len(redacted) > 50:
        print(f'{len(redacted) - 50} further findings; see the JSON report.')
    print(f'{len(scanner.public_collisions)} hash-bound public-origin marker collisions recorded')
    print(f"{len(problems)} problems in {root}")
    if args.report:
        args.report.write_text(json.dumps({'schema': 1, 'known_marker_count': len(deny),
            'coverage': dict(scanner.counts), 'problems': redacted,
            'public_collisions': scanner.public_collisions,
            'scope': __doc__.strip(), 'release_approved': False}, indent=2) + '\n')
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
