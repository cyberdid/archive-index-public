def _lat(p50, p95):
    return {"latency_ms": {"p50": p50, "p95": p95, "mean": p50, "n": 200}, "gpu_memory": {"added_mib": 12}}


def test_latency_table_has_both_devices_per_system():
    from paper.report.extra_tables import latency_table

    eff = {"queries": {"gpu": {"dense_eg2": _lat(112.34, 150.0), "peak_rss_mb": 4000},
                       "cpu": {"dense_eg2": _lat(214.7, 250.0), "peak_rss_mb": 4100}}}
    text = latency_table(eff, "en")
    assert "112.3" in text and "214.7" in text and "\\sys{dense\\_eg2}" in text
    assert "112.3" in latency_table(eff, "uk")


def test_quantization_quality_table_reports_difference_to_the_reference_model():
    from paper.report.extra_tables import gguf_table

    q = {"builds": {"Q8_0": {"file_mb": 300.0, "embed_chunks_per_s": 250.0,
                             "query_latency_ms_gpu_server": {"p50": 40.0}}},
         "quality": {"gemini_reviewed": {"systems": {"dense_eg2_text": {"nDCG@10": 0.631, "unjudged@10": 0.0,
                                                                        "nDCG@10_condensed": 0.631},
                                                      "gguf_Q8_0": {"nDCG@10": 0.629, "unjudged@10": 0.01,
                                                                    "nDCG@10_condensed": 0.633}},
                                         "comparisons": {"gguf_Q8_0": {"nDCG@10": {"mean": -0.002, "low": -0.01,
                                                                                   "high": 0.005, "p": 0.6},
                                                                      "nDCG@10_condensed": {"mean": 0.002, "low": -0.003,
                                                                                            "high": 0.007, "p": 0.4}}}}}}
    text = gguf_table(q, "en")
    assert "Q8\\_0" in text and "0.629" in text and "-0.002" in text and "300" in text
    assert "Condensed nDCG@10" in text and "0.633" in text and "0.002" in text and "0.400" in text


def test_ann_table_lists_each_collection_size():
    from paper.report.extra_tables import ann_table

    row = {"matrix_mb": 290.0, "numpy_exact_ms": {"p50": 5.0}, "faiss_flat_ms": {"p50": 3.0}, "hnsw_build_s": 20.0,
           "hnsw_ef64": {"recall@10": 0.97, "p50": 0.3}, "hnsw_ef256": {"recall@10": 0.995, "p50": 0.9}}
    text = ann_table({"sizes": {"100000": row, "1000000": row}}, "en")
    assert "100\\,000" in text and "1\\,000\\,000" in text and "0.970" in text


def test_control_table_compares_encoders_on_the_same_documents():
    from paper.report.extra_tables import control_table

    sys = lambda v: {"nDCG@10": v, "unjudged@10": 0.003}  # noqa: E731
    variant = {"systems": {"dense_e5": sys(0.674), "dense_eg2": sys(0.633), "dense_eg2_text": sys(0.631)},
               "comparisons": {"dense_eg2_text": {"nDCG@10": {"mean": -0.042, "low": -0.078, "high": -0.006, "p": 0.018}},
                               "dense_eg2": {"nDCG@10": {"mean": -0.041, "low": -0.077, "high": -0.005, "p": 0.023}}}}
    text = control_table({"variants": {"gemini_reviewed": variant, "gemma12_initial": variant}}, "en")
    assert "0.631" in text and "-0.042" in text and "0.018" in text


def test_judge_tables_show_known_answer_checks_and_pairwise_kappa():
    from paper.report.extra_tables import judge_agreement_table, judge_checks_table

    judge = {"graded": 5572, "mean_grade": 0.633, "distribution": {"0": 2554, "1": 2510, "2": 508},
             "target": {"pairs": 134, "share_2": 0.9179, "share_0": 0.0},
             "no_answer": {"pairs": 708, "share_0": 0.7571, "share_2": 0.0014}}
    report = {"judges": {"gpt-6-luna": judge},
              "pairs": [{"a": "gemini", "b": "gpt-6-luna", "n": 5572, "kappa": 0.621, "low": 0.595, "high": 0.644,
                         "exact_agreement": 0.759, "mean_a_minus_b": -0.212}]}
    checks = judge_checks_table(report, "en")
    assert "0.633" in checks and "91.8" in checks and "75.7" in checks and "GPT-6-Luna" in checks
    pairs = judge_agreement_table(report, "uk")
    assert "0.621 [0.595, 0.644]" in pairs and "-0.212" in pairs
