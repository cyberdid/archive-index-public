"""Booktabs fragments for the measurements added after the audit of 2026-10-22 (efficiency, quantization, ANN,
encoder control). Numbers are copied from the result JSON, never typed in; English and Ukrainian share them."""
from __future__ import annotations

from paper.report.tables import _table, tex_escape

SYSTEMS = ("bm25_plain", "bm25_lemma", "dense_eg2", "dense_eg2_256", "dense_e5", "rrf", "adaptive")
H = {
    "en": {"system": "System", "gpu50": "GPU p50", "gpu95": "GPU p95", "cpu50": "CPU p50", "cpu95": "CPU p95",
           "vram": "VRAM +MiB", "build": "Build", "mb": "File MiB", "cps": "Chunks/s", "q50": "Query p50",
           "ndcg": "nDCG@10", "diff": r"$\Delta$ [95\% CI]", "p": "p", "unj": "Unjudged@10", "size": "Vectors",
           "np": "NumPy exact", "flat": "Faiss flat", "hb": "HNSW build s", "h64": "HNSW ef64", "h256": "HNSW ef256",
           "ref": "bf16 (SentenceTransformers)", "judge": "Labels"},
    "uk": {"system": "Система", "gpu50": "GPU p50", "gpu95": "GPU p95", "cpu50": "CPU p50", "cpu95": "CPU p95",
           "vram": "VRAM +МіБ", "build": "Збірка", "mb": "Файл, МіБ", "cps": "Фрагм./с", "q50": "Запит p50",
           "ndcg": "nDCG@10", "diff": r"$\Delta$ [95\% ДІ]", "p": "p", "unj": "Неоцінені@10", "size": "Векторів",
           "np": "NumPy точний", "flat": "Faiss flat", "hb": "Побудова HNSW, с", "h64": "HNSW ef64",
           "h256": "HNSW ef256", "ref": "bf16 (SentenceTransformers)", "judge": "Мітки"},
}
VARIANT_NAMES = {"gemini_reviewed": "Gemini reviewed", "gemini_initial": "Gemini initial",
                 "gemma12_initial": "Gemma 12B initial", "median3": "Median of three"}


def _sys(name):
    return r"\sys{" + tex_escape(name) + "}"


def _int(n):
    return f"{int(n):,}".replace(",", r"\,")


def _diff(c):
    return (r"\shortstack{" + f"{c['mean']:+.3f}" + r" \\ {\footnotesize["
            + f"{c['low']:+.3f}, {c['high']:+.3f}" + r"]}}")


def latency_table(eff: dict, lang: str = "en") -> str:
    h, q = H[lang], eff["queries"]
    rows = []
    for name in SYSTEMS:
        if name in q.get("gpu", {}) and name in q.get("cpu", {}):
            g, c = q["gpu"][name], q["cpu"][name]
            rows.append([_sys(name), f"{g['latency_ms']['p50']:.1f}", f"{g['latency_ms']['p95']:.1f}",
                         f"{c['latency_ms']['p50']:.1f}", f"{c['latency_ms']['p95']:.1f}",
                         str(g["gpu_memory"]["added_mib"])])
    return _table([h["system"], h["gpu50"], h["gpu95"], h["cpu50"], h["cpu95"], h["vram"]], rows,
                  "per-query latency in ms (query embedding and chunk search; document collapse excluded); source efficiency.json")


def index_table(eff: dict, lang: str = "en") -> str:
    labels = {"en": ["Device", "Files", "Chunks", "Seconds", "Chunks/s", "DB MiB", "RSS MiB", "VRAM +MiB"],
              "uk": ["Пристрій", "Файли", "Фрагменти", "Секунди", "Фрагм./с", "БД, МіБ", "RSS, МіБ", "VRAM +МіБ"]}[lang]
    rows = []
    for device in ("gpu", "cpu"):
        r = eff.get("index", {}).get(device)
        if r:
            rows.append([device.upper(), str(r["files"]), _int(sum(r["chunks"].values())), f"{r['seconds']:.1f}",
                         f"{sum(r['chunks'].values()) / r['seconds']:.2f}", f"{r['disk_bytes'] / 2**20:.1f}",
                         f"{r['peak_rss_mb']:.0f}", str(r["gpu_memory"]["added_mib"])])
    text = _table(labels, rows, "indexing a fixed sample of the corpus with the archive CLI; source efficiency.json")
    full = next((r["full_index_disk_bytes"] for r in eff.get("index", {}).values() if r.get("full_index_disk_bytes")), {})
    if full:
        headers = ["Full index", "Main database MiB"] if lang == "en" else ["Повний індекс", "Основна БД, МіБ"]
        text += "\n\\par\\medskip\n" + _table(headers, [[tex_escape(n), f"{v / 2**20:.1f}"] for n, v in sorted(full.items())],
                                                 "main database files; sample disk above includes sidecars")
    return text


def field_table(field: dict, lang: str = "en") -> str:
    labels = {"en": ["Set", "Kind", "Queries", "Metric", "Mean [95\\% CI]"],
              "uk": ["Набір", "Тип", "Запитів", "Метрика", "Середнє [95\\% ДІ]"]}[lang]
    rows = []
    for name, kinds in field["sets"].items():
        for kind, values in kinds.items():
            for metric, value in values.items():
                if metric == "n":
                    continue
                rows.append([tex_escape(name), tex_escape(kind), str(values["n"]), tex_escape(metric),
                             f"{value['mean']:.3f} [{value['ci_low']:.3f}, {value['ci_high']:.3f}]"])
    archive = field.get("archive", {})
    summary = ((f"Archive: {_int(archive['documents'])} documents, {_int(archive['chunks'])} chunks."
                if lang == "en" else
                f"Архів: {_int(archive['documents'])} документів, {_int(archive['chunks'])} фрагментів.")
               + "\n\\par\\smallskip\n") if archive else ""
    return summary + _table(labels, rows, "private archive aggregates only; exact ranks are chunk ranks; source field_aggregate.json")


def gguf_table(quant: dict, lang: str = "en", variant: str = "gemini_reviewed") -> str:
    h = H[lang]
    quality = quant["quality"][variant]
    ref = quality["systems"]["dense_eg2_text"]
    rows = [[h["ref"], "--", "--", "--", f"{ref['nDCG@10']:.3f}", "--", "--", f"{ref['unjudged@10']:.3f}"]]
    for build, b in quant["builds"].items():
        s = quality["systems"][f"gguf_{build}"]
        c = quality["comparisons"][f"gguf_{build}"]["nDCG@10"]
        rows.append([tex_escape(build), f"{b['file_mb']:.0f}", f"{b['embed_chunks_per_s']:.0f}",
                     f"{b['query_latency_ms_gpu_server']['p50']:.1f}", f"{s['nDCG@10']:.3f}", _diff(c), f"{c['p']:.3f}",
                     f"{s['unjudged@10']:.3f}"])
    note = f"dense EG2 on the 4137 text documents; labels {variant}; source efficiency.json quantization"
    condensed_rows = [[h['ref'], f"{ref['nDCG@10_condensed']:.3f}", '--', '--']]
    for build in quant['builds']:
        name = f'gguf_{build}'
        stat = quality['comparisons'][name]['nDCG@10_condensed']
        condensed_rows.append([tex_escape(build), f"{quality['systems'][name]['nDCG@10_condensed']:.3f}",
                               _diff(stat), f"{stat['p']:.3f}"])
    return (_table([h["build"], h["mb"], h["cps"], h["q50"]], [r[:4] for r in rows], note)
            + "\n\\par\\medskip\n"
            + _table([h["build"], h["ndcg"], h["diff"], h["p"], h["unj"]],
                     [[r[0], *r[4:]] for r in rows], note)
            + "\n\\par\\medskip\n"
            + _table([h['build'], 'Condensed nDCG@10', h['diff'], h['p']], condensed_rows,
                     note + '; unjudged documents removed before ranking evaluation'))


def ann_table(ann: dict, lang: str = "en", reference: dict | None = None) -> str:
    h = H[lang]
    rows = []
    for n, r in sorted(ann["sizes"].items(), key=lambda kv: int(kv[0])):
        rows.append([_int(n), f"{r['numpy_exact_ms']['p50']:.2f}", f"{r['faiss_flat_ms']['p50']:.2f}",
                     f"{r['hnsw_build_s']:.0f}",
                     f"{r['hnsw_ef64']['p50']:.2f} / {r['hnsw_ef64']['recall@10']:.3f}",
                     f"{r['hnsw_ef256']['p50']:.2f} / {r['hnsw_ef256']['recall@10']:.3f}"])
    text = _table([h["size"], 'NumPy top-k', h["flat"], h["hb"], h["h64"], h["h256"]], rows,
                  "p50 ms per single query; HNSW cells: p50 ms / strict-ID recall@10 against exact; source ann_scaling.json")
    if reference and reference.get('complete'):
        labels = (['Vectors', 'Archive path p50 ms', 'Flat ID agreement', 'Flat score threshold'] if lang == 'en' else
                  ['Векторів', 'Пошук архіву p50, мс', 'Збіг ID flat', 'Поріг оцінки flat'])
        rows = [[_int(n), f"{r['archive_numpy_exact_ms']['p50']:.2f}",
                 f"{r['flat_strict_id_recall_vs_numpy']:.3f}",
                 f"{r['flat_results_above_reference_kth_score_with_tolerance']:.3f}"]
                for n, r in sorted(reference['sizes'].items(), key=lambda item: int(item[0]))]
        text += '\n\\par\\medskip\n' + _table(labels, rows,
                    'Actual semantic_scored path with candidate gather, full sort, pool400; no database polling; '
                    'flat exact agreement / fraction at reference kth score within1e-6; source ann_reference_audit.json')
        largest = max(map(int, reference['sizes']))
        actual = reference['sizes'][str(largest)]['archive_numpy_exact_ms']['p50']
        fast = ann['sizes'][str(largest)]['numpy_exact_ms']['p50']
        hnsw = ann['sizes'][str(largest)]['hnsw_ef256']
        if lang == 'en':
            text += ('\n\\par\\smallskip\n' + f'At {_int(largest)} vectors, observed p50 is {fast:.1f} ms for '
                     f'optimized NumPy top-k and {actual:.1f} ms for the archive vector path. HNSW ef256 is '
                     f"{hnsw['p50']:.2f} ms at strict-ID recall@10 {hnsw['recall@10']:.4f}. "
                     'These are sequential desktop-session measurements of different algorithms and output pools '
                     '(10 versus 400), not paired hardware speedups or end-to-end interface latency.\n')
        else:
            text += ('\n\\par\\smallskip\n' + f'На {_int(largest)} векторах спостережена p50 становить {fast:.1f} мс '
                     f'для оптимізованого NumPy top-k і {actual:.1f} мс для векторного пошуку архіву. HNSW ef256 має '
                     f"{hnsw['p50']:.2f} мс за повноти@10 за ідентифікаторами {hnsw['recall@10']:.4f}. "
                     'Це послідовні вимірювання різних алгоритмів і розмірів видачі (10 проти 400) у робочому сеансі, '
                     'а не парне прискорення обладнання чи наскрізна затримка інтерфейсу.\n')
    return text


def control_table(control: dict, lang: str = "en") -> str:
    h = H[lang]
    panels = []
    for variant, v in control["variants"].items():
        rows = []
        for name in ("dense_e5", "dense_eg2", "dense_eg2_text"):
            if name not in v["systems"]:
                continue
            s = v["systems"][name]
            c = v["comparisons"].get(name, {}).get("nDCG@10")
            rows.append([_sys(name), f"{s['nDCG@10']:.3f}",
                         _diff(c) if c else "--", f"{c['p']:.3f}" if c else "--", f"{s['unjudged@10']:.3f}"])
        panels.append(r"\noindent\textbf{" + tex_escape(VARIANT_NAMES.get(variant, variant)) + "}\n\\par\\smallskip\n"
                      + _table([h["system"], h["ndcg"], h["diff"] + " vs E5", h["p"], h["unj"]], rows,
                               "E5 against EG2 on the same 4137 text documents; source encoder_control.json"))
    return "\n\\par\\medskip\n".join(panels)


JUDGE_NAMES = {"gemini": "Gemini 3.8 Flash", "gemma4-12b": "Gemma 4 12B", "gemma4-e4b": "Gemma 4 E4B",
               "gpt-6-luna": "GPT-6-Luna", "claude": "Claude"}


def judge_checks_table(report: dict, lang: str = "en") -> str:
    labels = {"en": ["Judge", "Pairs", "Mean grade", "0 / 1 / 2", "Target = 2, \\%", "No-answer = 0, \\%"],
              "uk": ["Суддя", "Пари", "Середня оцінка", "0 / 1 / 2", "Ціль = 2, \\%", "Без відповіді = 0, \\%"]}[lang]
    rows = []
    for name, j in report["judges"].items():
        d = j["distribution"]
        rows.append([tex_escape(JUDGE_NAMES.get(name, name)), _int(j["graded"]), f"{j['mean_grade']:.3f}",
                     f"{_int(d['0'])} / {_int(d['1'])} / {_int(d['2'])}",
                     f"{100 * j['target']['share_2']:.1f} (n={j['target']['pairs']})",
                     f"{100 * j['no_answer']['share_0']:.1f} (n={j['no_answer']['pairs']})"])
    return _table(labels, rows, "grades on the whole pool and known-answer checks; source judges.json")


def judge_agreement_table(report: dict, lang: str = "en") -> str:
    labels = {"en": ["Judges", "Pairs", "Weighted $\\kappa$ [95\\% CI]", "Exact agreement", "Mean A$-$B"],
              "uk": ["Судді", "Пари", "Зважена $\\kappa$ [95\\% ДІ]", "Точний збіг", "Середня A$-$B"]}[lang]
    rows = [[tex_escape(f"{JUDGE_NAMES.get(p['a'], p['a'])} – {JUDGE_NAMES.get(p['b'], p['b'])}"), _int(p["n"]),
             f"{p['kappa']:.3f} [{p['low']:.3f}, {p['high']:.3f}]", f"{p['exact_agreement']:.3f}",
             f"{p['mean_a_minus_b']:+.3f}"] for p in report["pairs"]]
    return _table(labels, rows, "linear-weighted kappa, CI by resampling queries; source judges.json")
