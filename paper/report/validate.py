"""Validate source hashes, figure evidence, expected files and exact table numbers."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from paper.report.figures import figure_data, figure_names
from paper.report.generate import PLANNED, digest, load_sources, table_artifacts, unavailable_artifacts


def _metadata_errors(manifest, sources, reports, primary):
    """Check evidence claims separately from artifact bytes and renderer provenance."""
    provenance = reports[primary].get('provenance', {})
    status = ('complete within-model Gemini review' if primary == 'gemini_reviewed' else
              'initial judgments; review not available' if 'gemini_reviewed' not in reports else
              'initial judgments explicitly selected; reviewed sensitivity also available')
    expected = {
        'schema': 1,
        'status': status,
        'missing_planned_sources': [name for name in PLANNED if name not in sources],
        # These artifacts are outside the available-evidence renderer's supported scope.
        'unavailable_planned_artifacts': unavailable_artifacts(sources),
        'query_population': provenance.get('query_population', {}),
        'judging_provenance': provenance,
        'artifact_metadata': {
            'table_width': '160 mm linewidth; main/category split into panels',
            'table_intervals': '95% query bootstrap CIs from source; none fabricated',
            'figure_width_mm': 160, 'png_dpi': 300, 'pdf_fonttype': 42,
            'significance': 'paired p_holm<.05 against adaptive only',
            'R@100': 'pool-relative, incomplete judgments; not full gold'},
    }
    errors = [f'{field}: metadata disagrees with measured sources or generation contract'
              for field, value in expected.items() if manifest.get(field) != value]
    # Versions describe the generation environment, which may differ from the validation host.
    versions = manifest.get('software_versions')
    required = {'python', 'matplotlib', 'ir-measures', 'numpy'}
    pattern = (r'\d+\.\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?'
               r'(?:\+[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*)?')
    if (not isinstance(versions, dict) or not required <= versions.keys() or
            any(not isinstance(value, str) or re.fullmatch(pattern, value) is None
                for value in versions.values())):
        errors.append('software_versions: missing packages or invalid version values')
    return errors


def validate(results: Path, out: Path) -> list[str]:
    errors = []
    try:
        manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
        sources, reports, primary = load_sources(results, manifest['primary'])
        errors.extend(_metadata_errors(manifest, sources, reports, primary))
        expected_sources = {name: digest(results / name) for name in sources}
        if manifest['source_sha256'] != expected_sources: errors.append('source hash set or digest mismatch')
        texts = table_artifacts(sources, reports, primary)
        expected = set(texts)  # the fixed tables plus each optional table whose measured source exists
        chart_data = figure_data(reports, primary, sources.get('efficiency.json'), sources.get('ann_scaling.json'),
                                 sources.get('ann_reference_audit.json'))
        expected.update(f'figures/{name}.{ext}' for name in figure_names(chart_data) for ext in ('pdf', 'png'))
        if set(manifest['artifact_sha256']) != expected: errors.append('expected artifact set mismatch')
        for relative in expected:
            path = out / relative
            if not path.is_file():
                errors.append(f'missing artifact: {relative}')
                continue
            if digest(path) != manifest['artifact_sha256'].get(relative): errors.append(f'artifact hash mismatch: {relative}')
            if relative in texts and path.read_text(encoding='utf-8') != texts[relative]:
                errors.append(f'numeric/text evidence mismatch: {relative}')
            if relative.endswith('.pdf') and not path.read_bytes().startswith(b'%PDF'):
                errors.append(f'invalid PDF: {relative}')
            if relative.endswith('.png'):
                from PIL import Image
                with Image.open(path) as img:
                    if img.width < 1800 or min(img.info.get('dpi', (0, 0))) < 299:
                        errors.append(f'PNG resolution below 300 dpi at 160 mm: {relative}')
        if manifest['figure_data'] != chart_data: errors.append('figure numeric evidence mismatch')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f'validation failed: {exc}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('paper/results'))
    parser.add_argument('--out', type=Path, default=Path('paper/manuscript/generated'))
    args = parser.parse_args()
    errors = validate(args.results, args.out)
    if errors:
        print('\n'.join(errors))
        return 1
    print('Validated source hashes, publication numbers, figure evidence and expected artifacts.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
