import numpy as np
import pytest


def test_scaled_collection_keeps_unit_norm_and_contains_the_real_vectors():
    from paper.eval.ann_scaling import scaled_collection

    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 16)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    big = scaled_collection(base, 180, sigma=0.05, seed=1)
    assert big.shape == (180, 16) and big.dtype == np.float32
    assert np.allclose(np.linalg.norm(big, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(big[:50], base)  # the real vectors come first, unchanged
    assert np.array_equal(big, scaled_collection(base, 180, sigma=0.05, seed=1))


def test_recall_at_k_counts_shared_neighbours():
    from paper.eval.ann_scaling import recall_at_k

    exact = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    approx = np.array([[4, 3, 9, 1], [5, 0, 0, 0]])
    assert recall_at_k(exact, approx, k=4) == pytest.approx((3 / 4 + 1 / 4) / 2)


def test_exact_search_returns_the_best_inner_products():
    from paper.eval.ann_scaling import exact_top_k

    matrix = np.eye(5, dtype=np.float32)
    queries = np.array([[0.1, 0.9, 0, 0, 0.5]], dtype=np.float32)
    assert exact_top_k(matrix, queries, k=2).tolist() == [[1, 4]]
