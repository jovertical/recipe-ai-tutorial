# Problem 6: Low-Rank Matrix Decomposition
#
# Learn the math behind LoRA: approximating matrices with low-rank factors.
#
# Key insight:
# - A weight matrix W has shape (d_in, d_out) with d_in * d_out parameters
# - We can approximate W ≈ B @ A where:
#   - A has shape (r, d_out) with r * d_out parameters
#   - B has shape (d_in, r) with d_in * r parameters
#   - r << min(d_in, d_out) is the "rank"
# - Total params: r * (d_in + d_out) << d_in * d_out
#
# Example:
#   W: (1000, 1000) = 1,000,000 parameters
#   B @ A: (1000, 8) @ (8, 1000) = 8,000 + 8,000 = 16,000 parameters
#   Compression: 62.5x fewer parameters!
#
# ML Relevance: This is THE core idea of LoRA. Instead of updating a huge
# weight matrix, we learn small low-rank factors that modify it.

import numpy as np


def low_rank_approximation(W: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate matrix W with low-rank factors using SVD.

    Args:
        W: Matrix to approximate, shape (m, n)
        rank: Target rank (r)

    Returns:
        Tuple of (B, A) where W ≈ B @ A
        - B: shape (m, rank)
        - A: shape (rank, n)

    Use SVD: W = U @ S @ V^T
    Low-rank: W_r = U[:, :r] @ S[:r, :r] @ V[:r, :]
    Factor: B = U[:, :r] @ sqrt(S[:r, :r])
            A = sqrt(S[:r, :r]) @ V[:r, :]
    """
    # Your solution here
    pass


def reconstruction_error(W: np.ndarray, B: np.ndarray, A: np.ndarray) -> float:
    """
    Compute reconstruction error (Frobenius norm).

    Error = ||W - B @ A||_F / ||W||_F

    Returns relative error (0 = perfect, 1 = terrible)
    """
    # Your solution here
    pass


def rank_vs_accuracy(W: np.ndarray, ranks: list[int]) -> dict:
    """
    Analyze trade-off between rank and approximation accuracy.

    Args:
        W: Original matrix
        ranks: List of ranks to try

    Returns:
        Dict with:
        - 'ranks': input ranks
        - 'errors': reconstruction error for each rank
        - 'compression': compression ratio for each rank
        - 'params_original': original parameter count
        - 'params_per_rank': parameter count for each rank
    """
    # Your solution here
    pass


def optimal_rank_for_error(W: np.ndarray, max_error: float) -> int:
    """
    Find minimum rank that achieves target error.

    Args:
        W: Original matrix
        max_error: Maximum acceptable relative error

    Returns:
        Minimum rank r such that reconstruction error < max_error
    """
    # Your solution here
    pass


def effective_rank(W: np.ndarray, threshold: float = 0.99) -> int:
    """
    Compute effective rank of a matrix.

    Effective rank = number of singular values needed to capture
    'threshold' fraction of total singular value mass.

    This tells you how "low-rank" a matrix naturally is.
    Many neural network weight matrices have low effective rank!
    """
    # Your solution here
    pass


def random_low_rank_matrix(m: int, n: int, rank: int) -> np.ndarray:
    """
    Create a random matrix that is exactly low-rank.

    Useful for testing: if we create a rank-r matrix,
    low_rank_approximation with r should recover it exactly.
    """
    # Your solution here
    pass


def compare_random_init_vs_svd(
    W: np.ndarray,
    rank: int,
    num_trials: int = 10
) -> dict:
    """
    Compare SVD initialization vs random initialization for low-rank factors.

    In LoRA, factors are initialized specially:
    - A: Random Gaussian / sqrt(r)
    - B: Zeros (so initial LoRA contribution is 0)

    Compare with SVD-based initialization.

    Returns:
        - 'svd_error': Error with SVD factors
        - 'random_errors': List of errors with random factors
        - 'random_mean': Mean random error
        - 'random_std': Std of random errors
    """
    # Your solution here
    pass


def visualize_rank_tradeoff(W: np.ndarray) -> str:
    """
    Create a text visualization of rank vs accuracy trade-off.

    Show:
    - Error curve as rank increases
    - Where "knee" of curve is (diminishing returns)
    - Compression ratios
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: Low-rank approximation shapes
    W = np.random.randn(50, 30)
    B, A = low_rank_approximation(W, rank=5)
    assert B.shape == (50, 5), f"Test 1a failed: expected B shape (50, 5), got {B.shape}"
    assert A.shape == (5, 30), f"Test 1b failed: expected A shape (5, 30), got {A.shape}"

    # Test 2: Reconstruction
    W_approx = B @ A
    assert W_approx.shape == W.shape, "Test 2 failed: reconstructed shape should match"

    # Test 3: Reconstruction error
    error = reconstruction_error(W, B, A)
    assert 0 <= error <= 1, f"Test 3a failed: error {error} should be in [0, 1]"
    # With rank 5 out of 30, should have some error
    assert error > 0, "Test 3b failed: rank-5 approx of random matrix should have error"

    # Test 4: Higher rank = lower error
    B_high, A_high = low_rank_approximation(W, rank=20)
    error_high = reconstruction_error(W, B_high, A_high)
    assert error_high < error, "Test 4 failed: higher rank should have lower error"

    # Test 5: Full rank = zero error
    B_full, A_full = low_rank_approximation(W, rank=30)  # Full rank
    error_full = reconstruction_error(W, B_full, A_full)
    assert error_full < 1e-10, f"Test 5 failed: full rank should have ~0 error, got {error_full}"

    # Test 6: Rank vs accuracy analysis
    analysis = rank_vs_accuracy(W, ranks=[1, 5, 10, 20, 30])
    assert len(analysis['errors']) == 5, "Test 6a failed: should have 5 error values"
    assert all(analysis['errors'][i] >= analysis['errors'][i+1] for i in range(4)), \
        "Test 6b failed: errors should decrease with rank"

    # Test 7: Compression ratio
    # Rank 5: (50*5 + 5*30) = 400 params vs 50*30 = 1500 original
    assert analysis['compression'][1] > 3, "Test 7 failed: rank 5 should compress >3x"

    # Test 8: Optimal rank for error
    opt_rank = optimal_rank_for_error(W, max_error=0.1)
    B_opt, A_opt = low_rank_approximation(W, rank=opt_rank)
    actual_error = reconstruction_error(W, B_opt, A_opt)
    assert actual_error < 0.1, f"Test 8 failed: error {actual_error} should be < 0.1"

    # Test 9: Effective rank
    # A truly low-rank matrix should have low effective rank
    W_lowrank = random_low_rank_matrix(50, 30, rank=3)
    eff_rank = effective_rank(W_lowrank)
    assert eff_rank <= 5, f"Test 9 failed: rank-3 matrix should have effective rank ~3, got {eff_rank}"

    # Test 10: Random low-rank matrix
    W_test = random_low_rank_matrix(40, 25, rank=7)
    assert W_test.shape == (40, 25), "Test 10a failed: wrong shape"
    # SVD should recover it with rank 7
    B_rec, A_rec = low_rank_approximation(W_test, rank=7)
    error_rec = reconstruction_error(W_test, B_rec, A_rec)
    assert error_rec < 1e-10, f"Test 10b failed: should perfectly recover, got error {error_rec}"

    # Test 11: Compare initialization strategies
    comparison = compare_random_init_vs_svd(W, rank=10)
    assert 'svd_error' in comparison, "Test 11a failed: missing svd_error"
    assert 'random_mean' in comparison, "Test 11b failed: missing random_mean"
    # SVD should be better than random
    assert comparison['svd_error'] < comparison['random_mean'], \
        "Test 11c failed: SVD should be better than random init"

    # Test 12: Visualization
    viz = visualize_rank_tradeoff(W)
    assert isinstance(viz, str), "Test 12a failed: should return string"
    assert len(viz) > 0, "Test 12b failed: should not be empty"

    print("All tests passed!")
    print("\nLow-rank approximation is the mathematical foundation of LoRA.")
    print("Instead of updating W directly, LoRA learns W + B @ A with small r.")
