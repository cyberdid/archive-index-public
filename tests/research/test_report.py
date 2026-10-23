"""Publication checks use hand-specified values, not a formatter copied into a test."""
import copy
import json
import re

import pytest

METRICS = ('nDCG@10', 'R@100', 'RR@10', 'Success@1', 'Success@5')
SYSTEM_MEANS = {'adaptive': .6, 'bm25_plain': .3, 'bm25_lemma': .4, 'dense_eg2': .5, 'dense_e5': .7,
                'rrf': .52, 'adaptive-ident': .59, 'adaptive-phrase': .6, 'adaptive-short': .55, 'dense_eg2_256': .48}
CATEGORIES = ('known_item', 'topical', 'technical', 'cross_lingual')


@pytest.fixture
def report():
    def measures(mean, n=4):
        return {m: {'mean': mean, 'ci_low': mean-.1, 'ci_high': mean+.1, 'n': n} for m in METRICS}
    return {'variant': 'gemini_initial', 'queries': 4,
            'provenance': {'reference_system': 'adaptive', 'seed': 2026, 'bootstrap_resamples': 10000,
                           'query_population': {'total': 5, 'answerable': 4, 'no_answer': 1},
                           'input_sha256': {'queries.tsv': 'a'*64}},
            'systems': {s: {'all': measures(v), 'by_category': {c: measures(v, 1) for c in CATEGORIES},
                            'by_language': {l: measures(v, 2) for l in ('uk', 'ru')}}
                        for s, v in SYSTEM_MEANS.items()},
            'comparisons': {m: [{'system': 'bm25_lemma', 'p_holm': .049},
                               {'system': 'dense_e5', 'p_holm': .057},
                               {'system': 'bm25_plain', 'p_holm': .05}] for m in METRICS}}


@pytest.fixture
def results(tmp_path, report):
    root = tmp_path / 'results'
    root.mkdir()
    secondary = copy.deepcopy(report)
    secondary['variant'] = 'gemma12_initial'
    sources = {'ua_par_gemini_initial.json': report, 'ua_par_gemma12_initial.json': secondary,
               'judge_sensitivity.json': {'means': {v: {m: {s: val for s, val in SYSTEM_MEANS.items()}
                                                        for m in METRICS}
                                                   for v in ('gemini_initial', 'gemma12_initial')}},
               'judges.json': {'judges': {'gemini': {'graded': 20}, 'gemma4-12b': {'graded': 20}}},
               'multimodal.json': {'images': {'uk': {'recall@1': .75, 'recall@5': 1., 'recall@10': 1.,
                                                    'mrr': .8, 'queries': 4, 'items': 4}}, 'speech': {}},
               'corpus_summary.json': {'manifest_files': 10, 'queries': 5,
                                       'queries_by_category': {'no_answer': 1}, 'target_parts': {'notes': 4},
                                       'indexes': {'full': {'documents': 8}, 'e5': {'documents': 6}}},
               'quantization_reference.json': {'Q8_0': {'mean': .99, 'worst': .98, 'n': 20,
                                                        'file_mb': 295.5, 'cpu_s_per_text': .069}}}
    for name, value in sources.items(): (root / name).write_text(json.dumps(value))
    return root


def test_main_table_marks_only_existing_paired_holm_evidence(report):
    from paper.report.tables import main_table
    table = main_table(report)
    assert '0.400' in table and '[0.300, 0.500]' in table
    assert 'p_{\\mathrm{Holm}}<0.05' in table
    assert 'pool-relative' in table
    assert 'resizebox' not in table and 'tiny' not in table
    # There are 5 significant cells for lemma; p=.050 and .057 do not receive markers.
    assert table.count(r'\textsuperscript{\textdagger}') == 5  # exactly five data markers; legend is separate
    assert '0.700\\textsuperscript' not in table
    # Different overlapping marginal intervals cannot suppress actual paired evidence.
    report['systems']['bm25_lemma']['all']['nDCG@10']['ci_high'] = .65
    assert '0.400\\textsuperscript' in main_table(report)


def test_tables_escape_labels_and_share_exact_numbers_across_languages(report):
    from paper.report.tables import main_table, tex_escape
    report['systems']['x_#%&{}'] = report['systems'].pop('rrf')
    en, uk = main_table(report, 'en'), main_table(report, 'uk')
    assert r'x\_\#\%\&\{\}' in en
    assert re.findall(r'\d+\.\d{3}', en) == re.findall(r'\d+\.\d{3}', uk)
    assert tex_escape('a\\b~^$') == r'a\textbackslash{}b\textasciitilde{}\textasciicircum{}\$'


def test_multimodal_table_does_not_fabricate_confidence_intervals():
    from paper.report.tables import multimodal_table
    table = multimodal_table({'images': {'uk': {'recall@1': .75, 'recall@5': 1., 'recall@10': 1.,
                                               'mrr': .8, 'queries': 4, 'items': 4}}, 'speech': {}})
    assert '0.750' in table and '0.800' in table
    assert 'unavailable' in table and '95' not in table


def test_generate_manifest_tracks_sources_values_missing_evidence_and_validates(results, tmp_path):
    from paper.report.generate import generate
    from paper.report.validate import validate
    out = tmp_path / 'generated'
    manifest = generate(results, out)
    assert manifest['primary'] == 'gemini_initial'
    assert manifest['status'] == 'initial judgments; review not available'
    assert 'efficiency.json' in manifest['missing_planned_sources']
    assert 'field_aggregate.json' in manifest['missing_planned_sources']
    assert not (out / 'en/efficiency.tex').exists()
    assert manifest['source_sha256']['ua_par_gemini_initial.json']
    assert manifest['figure_data']['category']['rows'][0]['mean'] in SYSTEM_MEANS.values()
    assert (out / 'figures/category_ndcg.pdf').read_bytes().startswith(b'%PDF')
    assert (out / 'figures/judge_sensitivity.png').read_bytes().startswith(b'\x89PNG')
    assert validate(results, out) == []
    # The independently specified value appears in the publication fragment.
    assert '0.400' in (out / 'en/main.tex').read_text()
    original = (out / 'en/main.tex').read_text()
    (out / 'en/main.tex').write_text(original.replace('0.400', '0.999', 1))
    assert any('main.tex' in error for error in validate(results, out))


def test_validation_rejects_changed_source_even_if_generated_files_are_untouched(results, tmp_path):
    from paper.report.generate import generate
    from paper.report.validate import validate
    out = tmp_path / 'generated'
    generate(results, out)
    source = results / 'multimodal.json'
    source.write_text(source.read_text() + '\n')
    assert any('source hash' in error for error in validate(results, out))


@pytest.mark.parametrize('mutation', ['nonfinite', 'bad_ci', 'missing_ci', 'missing_source', 'wrong_sensitivity'])
def test_generation_fails_closed_on_incomplete_or_invalid_measured_evidence(results, tmp_path, mutation):
    from paper.report.generate import generate
    path = results / 'ua_par_gemini_initial.json'
    data = json.loads(path.read_text())
    stat = data['systems']['adaptive']['all']['nDCG@10']
    if mutation == 'nonfinite': stat['mean'] = float('nan')
    elif mutation == 'bad_ci': stat['ci_low'] = .9
    elif mutation == 'missing_ci': del stat['ci_high']
    elif mutation == 'missing_source': (results / 'multimodal.json').unlink()
    elif mutation == 'wrong_sensitivity':
        source = results / 'judge_sensitivity.json'
        value = json.loads(source.read_text())
        value['means']['gemini_initial']['nDCG@10']['adaptive'] = .999
        source.write_text(json.dumps(value))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): generate(results, tmp_path / 'generated')


def test_reviewed_data_becomes_primary_only_when_present_and_consistent(results, tmp_path):
    from paper.report.generate import generate
    initial = json.loads((results / 'ua_par_gemini_initial.json').read_text())
    initial['variant'] = 'gemini_reviewed'
    (results / 'ua_par_gemini_reviewed.json').write_text(json.dumps(initial))
    sensitivity = json.loads((results / 'judge_sensitivity.json').read_text())
    sensitivity['means']['gemini_reviewed'] = sensitivity['means']['gemini_initial']
    (results / 'judge_sensitivity.json').write_text(json.dumps(sensitivity))
    manifest = generate(results, tmp_path / 'generated')
    assert manifest['primary'] == 'gemini_reviewed'
    assert 'within-model' in manifest['status']
    assert set(manifest['figure_data']['sensitivity']['variants']) == {'gemini_initial', 'gemma12_initial', 'gemini_reviewed'}


def test_manifest_records_actual_renderer_versions(results, tmp_path):
    from importlib.metadata import version
    from paper.report.generate import generate
    manifest = generate(results, tmp_path / 'generated')
    assert manifest['software_versions']['matplotlib'] == version('matplotlib')
    assert manifest['software_versions']['ir-measures'] == version('ir-measures')
    assert manifest['software_versions']['python']


def test_category_counts_and_ukrainian_reader_facing_notes(results, tmp_path):
    from paper.report.generate import generate
    out = tmp_path / 'generated'
    manifest = generate(results, out)
    assert manifest['figure_data']['category']['counts'] == {c: 1 for c in CATEGORIES}
    assert 'Зібрані файли' in (out / 'uk/corpus.tex').read_text()
    assert 'Середній косинус' in (out / 'uk/quantization_reference.tex').read_text()
    notes = (out / 'uk/evidence_notes.tex').read_text()
    assert 'graded pairs' not in notes and 'initial; review' not in notes
    assert 'оцінених пар' in notes


@pytest.mark.parametrize('field,value', [
    ('status', 'complete within-model Gemini review'),
    ('missing_planned_sources', []),
    ('unavailable_planned_artifacts', []),
    ('query_population', {'total': 999, 'answerable': 999, 'no_answer': 0}),
    ('judging_provenance', {'seed': 0}),
    ('software_versions', {'python': 'not a version', 'matplotlib': 'unknown', 'ir-measures': None, 'numpy': 1}),
    ('software_versions', {'python': '3.13.14'}),
    ('artifact_metadata', {'R@100': 'complete gold judgments'}),
])
def test_validator_refuses_false_manifest_evidence_claims(results, tmp_path, field, value):
    from paper.report.generate import generate
    from paper.report.validate import validate
    out = tmp_path / 'generated'
    manifest = generate(results, out)
    manifest[field] = value
    (out / 'manifest.json').write_text(json.dumps(manifest))
    errors = validate(results, out)
    assert any(field in error for error in errors), errors


def test_validator_accepts_valid_software_provenance_from_another_host(results, tmp_path):
    from paper.report.generate import generate
    from paper.report.validate import validate
    out = tmp_path / 'generated'
    manifest = generate(results, out)
    manifest['software_versions'] = {'python': '3.12.8', 'matplotlib': '3.10.8', 'ir-measures': '0.4.2', 'numpy': '2.2.0'}
    (out / 'manifest.json').write_text(json.dumps(manifest))
    assert validate(results, out) == []
