# Problem 2: Scaled Dot-Product Attention
#
# Add scaling to prevent softmax saturation.
#
# The problem with unscaled attention:
# - As dimension d_k grows, dot products grow in magnitude
# - Large dot products push softmax into saturation (near 0 or 1)
# - Saturated softmax has tiny gradients (vanishing gradient problem)
#
# The solution:
#   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
#
# Why sqrt(d_k)?
# - If Q and K have elements with mean=0, variance=1
# - Then QK^T has elements with variance=d_k
# - Dividing by sqrt(d_k) brings variance back to 1
#
# ML Relevance: This is a crucial numerical stability trick. Understanding
# why we scale helps you debug gradient issues in deep networks.

import numpy as np


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    Args:
        query: Shape (seq_len_q, d_k)
        key: Shape (seq_len_k, d_k)
        value: Shape (seq_len_k, d_v)

    Returns:
        Tuple of:
        - output: Shape (seq_len_q, d_v)
        - weights: Shape (seq_len_q, seq_len_k)

    Formula: softmax(QK^T / sqrt(d_k)) @ V
    """
    # Your solution here
    pass


def demonstrate_gradient_problem(d_k_values: list[int]) -> dict:
    """
    Demonstrate why scaling is necessary.

    For different dimension sizes, compute:
    1. Unscaled attention scores
    2. Softmax outputs
    3. Softmax gradients

    Returns dict with:
    - 'd_k': list of dimensions tested
    - 'unscaled_variance': variance of unscaled QK^T
    - 'scaled_variance': variance of scaled QK^T / sqrt(d_k)
    - 'unscaled_max_softmax': max softmax value (1.0 = saturated)
    - 'scaled_max_softmax': max softmax value with scaling
    - 'unscaled_grad_norm': gradient norm without scaling
    - 'scaled_grad_norm': gradient norm with scaling
    """
    # Your solution here
    pass


def softmax_gradient(softmax_output: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of softmax with respect to its input.

    For softmax output s, the Jacobian is:
    diag(s) - s @ s^T

    Returns the diagonal elements of the Jacobian (simplified gradient).
    The gradient is s * (1 - s) for each element.

    When softmax is saturated (s near 0 or 1), gradient is near 0.
    """
    # Your solution here
    pass


def compare_scaled_vs_unscaled(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray
) -> dict:
    """
    Compare attention with and without scaling.

    Returns:
    - 'unscaled_weights': attention weights without scaling
    - 'scaled_weights': attention weights with scaling
    - 'unscaled_output': output without scaling
    - 'scaled_output': output with scaling
    - 'weight_difference': L2 norm of weight difference
    - 'entropy_unscaled': entropy of unscaled weights (lower = more focused)
    - 'entropy_scaled': entropy of scaled weights
    """
    # Your solution here
    pass


def entropy(probs: np.ndarray) -> float:
    """
    Compute entropy of a probability distribution.

    Entropy = -sum(p * log(p))

    Higher entropy = more uniform distribution
    Lower entropy = more peaked distribution

    Handle p=0 by treating 0*log(0) as 0.
    """
    # Your solution here
    pass


def visualize_scaling_effect(d_k: int) -> str:
    """
    Create a visualization showing the effect of scaling.

    Generate random Q, K and show:
    1. Raw score distribution
    2. Unscaled softmax
    3. Scaled softmax

    Returns a text representation.
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Basic scaled attention
    Q = np.array([[1.0, 0.0, 1.0, 0.0]])
    K = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    V = np.array([[1.0, 2.0], [3.0, 4.0]])

    output, weights = scaled_dot_product_attention(Q, K, V)
    assert output.shape == (1, 2), f"Test 1a failed: expected (1, 2), got {output.shape}"
    assert abs(weights.sum() - 1.0) < 1e-6, "Test 1b failed: weights should sum to 1"

    # Test 2: Scaling reduces score magnitude
    d_k = 64
    Q_large = np.random.randn(1, d_k)
    K_large = np.random.randn(10, d_k)
    V_large = np.random.randn(10, 32)

    # Unscaled scores have high variance
    unscaled_scores = Q_large @ K_large.T
    scaled_scores = unscaled_scores / np.sqrt(d_k)

    assert scaled_scores.var() < unscaled_scores.var(), "Test 2 failed: scaling should reduce variance"

    # Test 3: Softmax gradient
    probs = np.array([0.2, 0.5, 0.3])
    grad = softmax_gradient(probs)
    assert len(grad) == 3, f"Test 3a failed: expected 3 gradients, got {len(grad)}"
    # Gradient should be s*(1-s)
    expected = probs * (1 - probs)
    assert np.allclose(grad, expected), f"Test 3b failed: expected {expected}, got {grad}"

    # Test 4: Saturated softmax has small gradient
    saturated_probs = np.array([0.99, 0.005, 0.005])
    saturated_grad = softmax_gradient(saturated_probs)
    uniform_probs = np.array([0.33, 0.34, 0.33])
    uniform_grad = softmax_gradient(uniform_probs)
    assert saturated_grad.max() < uniform_grad.max(), \
        "Test 4 failed: saturated softmax should have smaller gradient"

    # Test 5: Entropy calculation
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    assert entropy(uniform) > entropy(peaked), "Test 5 failed: uniform should have higher entropy"

    # Test 6: Entropy of uniform distribution
    n = 4
    uniform = np.ones(n) / n
    expected_entropy = np.log(n)  # Maximum entropy for n outcomes
    assert abs(entropy(uniform) - expected_entropy) < 1e-6, \
        f"Test 6 failed: expected {expected_entropy}, got {entropy(uniform)}"

    # Test 7: Compare scaled vs unscaled
    comparison = compare_scaled_vs_unscaled(Q, K, V)
    assert 'unscaled_weights' in comparison, "Test 7a failed: missing unscaled_weights"
    assert 'scaled_weights' in comparison, "Test 7b failed: missing scaled_weights"
    assert 'entropy_unscaled' in comparison, "Test 7c failed: missing entropy_unscaled"
    assert 'entropy_scaled' in comparison, "Test 7d failed: missing entropy_scaled"

    # Test 8: Scaling increases entropy (less peaked)
    # For high-dimensional inputs, scaling should make weights more uniform
    d_k = 512
    Q_high = np.random.randn(1, d_k) * 2  # Higher variance
    K_high = np.random.randn(20, d_k) * 2
    V_high = np.random.randn(20, 16)

    comparison = compare_scaled_vs_unscaled(Q_high, K_high, V_high)
    assert comparison['entropy_scaled'] >= comparison['entropy_unscaled'] - 0.1, \
        "Test 8 failed: scaling should generally increase entropy"

    # Test 9: Demonstrate gradient problem
    demo = demonstrate_gradient_problem([8, 64, 512])
    assert len(demo['d_k']) == 3, "Test 9a failed: should test 3 dimensions"
    assert 'unscaled_variance' in demo, "Test 9b failed: missing unscaled_variance"
    assert 'scaled_variance' in demo, "Test 9c failed: missing scaled_variance"

    # Test 10: Visualization
    viz = visualize_scaling_effect(64)
    assert isinstance(viz, str), "Test 10a failed: should return string"
    assert len(viz) > 0, "Test 10b failed: visualization should not be empty"

    print("All tests passed!")
    print("\nKey insight: Without scaling, high-dimensional attention saturates softmax,")
    print("causing vanishing gradients. Dividing by sqrt(d_k) keeps gradients healthy.")
