# Problem 1: Dot Product Attention
#
# Implement the core attention operation from scratch.
# This is the fundamental building block of all transformer models.
#
# The attention formula:
#   Attention(Q, K, V) = softmax(QK^T) * V
#
# Where:
#   Q (Query): What we're looking for
#   K (Key): What each position offers
#   V (Value): The actual content to retrieve
#
# Example:
#   Q = [[1, 0]]       # 1 query of dimension 2
#   K = [[1, 0],       # 2 keys of dimension 2
#        [0, 1]]
#   V = [[1, 2],       # 2 values of dimension 2
#        [3, 4]]
#
#   Scores = Q @ K^T = [[1, 0]]  # Query attends to first key
#   Weights = softmax([[1, 0]]) = [[0.73, 0.27]]
#   Output = Weights @ V = [[1.54, 2.54]]
#
# ML Relevance: This is THE core operation in transformers. Every GPT, BERT,
# and Llama model uses this hundreds of times per forward pass.

import numpy as np


def dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute dot-product attention.

    Args:
        query: Shape (seq_len_q, d_k) - the queries
        key: Shape (seq_len_k, d_k) - the keys
        value: Shape (seq_len_k, d_v) - the values

    Returns:
        Tuple of:
        - output: Shape (seq_len_q, d_v) - attention output
        - weights: Shape (seq_len_q, seq_len_k) - attention weights

    Steps:
    1. Compute attention scores: Q @ K^T
    2. Apply softmax to get weights
    3. Compute weighted sum of values
    """
    # Your solution here
    pass


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        Softmax probabilities (same shape as input)
    """
    # Your solution here
    # Hint: Subtract max for numerical stability
    pass


def attention_weights(
    query: np.ndarray,
    key: np.ndarray
) -> np.ndarray:
    """
    Compute just the attention weights (useful for visualization).

    Args:
        query: Shape (seq_len_q, d_k)
        key: Shape (seq_len_k, d_k)

    Returns:
        weights: Shape (seq_len_q, seq_len_k)
    """
    # Your solution here
    pass


def visualize_attention(
    weights: np.ndarray,
    query_labels: list[str],
    key_labels: list[str]
) -> str:
    """
    Create a text visualization of attention weights.

    Args:
        weights: Shape (seq_len_q, seq_len_k)
        query_labels: Labels for query positions
        key_labels: Labels for key positions

    Returns:
        String representation of attention matrix

    Example output:
           | the | cat | sat |
    -------+-----+-----+-----+
    what   | 0.1 | 0.8 | 0.1 |
    is     | 0.3 | 0.4 | 0.3 |
    """
    # Your solution here
    pass


def batch_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batched version of dot-product attention.

    Args:
        query: Shape (batch_size, seq_len_q, d_k)
        key: Shape (batch_size, seq_len_k, d_k)
        value: Shape (batch_size, seq_len_k, d_v)

    Returns:
        Tuple of:
        - output: Shape (batch_size, seq_len_q, d_v)
        - weights: Shape (batch_size, seq_len_q, seq_len_k)
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Softmax
    x = np.array([1.0, 2.0, 3.0])
    s = softmax(x)
    assert abs(s.sum() - 1.0) < 1e-6, "Test 1a failed: softmax should sum to 1"
    assert s[2] > s[1] > s[0], "Test 1b failed: softmax should preserve order"

    # Test 2: Softmax numerical stability
    x_large = np.array([1000.0, 1001.0, 1002.0])
    s_large = softmax(x_large)
    assert not np.isnan(s_large).any(), "Test 2 failed: softmax should be numerically stable"

    # Test 3: Softmax 2D
    x_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    s_2d = softmax(x_2d, axis=-1)
    assert np.allclose(s_2d.sum(axis=-1), [1.0, 1.0]), "Test 3 failed: 2D softmax should sum to 1 per row"

    # Test 4: Basic attention
    Q = np.array([[1.0, 0.0]])  # 1 query
    K = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 keys
    V = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 values

    output, weights = dot_product_attention(Q, K, V)
    assert output.shape == (1, 2), f"Test 4a failed: expected shape (1, 2), got {output.shape}"
    assert weights.shape == (1, 2), f"Test 4b failed: expected shape (1, 2), got {weights.shape}"
    assert abs(weights.sum() - 1.0) < 1e-6, "Test 4c failed: attention weights should sum to 1"

    # Test 5: Attention focuses on matching key
    # Query [1, 0] should attend more to key [1, 0] than key [0, 1]
    assert weights[0, 0] > weights[0, 1], "Test 5 failed: attention should focus on matching key"

    # Test 6: Attention weights
    weights_only = attention_weights(Q, K)
    assert np.allclose(weights, weights_only), "Test 6 failed: attention_weights should match"

    # Test 7: Multiple queries
    Q_multi = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 queries
    output_multi, weights_multi = dot_product_attention(Q_multi, K, V)
    assert output_multi.shape == (2, 2), f"Test 7a failed: expected (2, 2), got {output_multi.shape}"
    assert weights_multi.shape == (2, 2), f"Test 7b failed: expected (2, 2), got {weights_multi.shape}"

    # Test 8: Visualization
    viz = visualize_attention(
        weights_multi,
        query_labels=["q1", "q2"],
        key_labels=["k1", "k2"]
    )
    assert isinstance(viz, str), "Test 8a failed: should return string"
    assert len(viz) > 0, "Test 8b failed: visualization should not be empty"

    # Test 9: Batch attention
    batch_size = 3
    Q_batch = np.random.randn(batch_size, 4, 8)  # 3 batches, 4 queries, dim 8
    K_batch = np.random.randn(batch_size, 6, 8)  # 3 batches, 6 keys, dim 8
    V_batch = np.random.randn(batch_size, 6, 16)  # 3 batches, 6 values, dim 16

    output_batch, weights_batch = batch_dot_product_attention(Q_batch, K_batch, V_batch)
    assert output_batch.shape == (3, 4, 16), f"Test 9a failed: expected (3, 4, 16), got {output_batch.shape}"
    assert weights_batch.shape == (3, 4, 6), f"Test 9b failed: expected (3, 4, 6), got {weights_batch.shape}"

    # Test 10: Uniform attention with equal queries
    Q_uniform = np.ones((1, 4))
    K_uniform = np.ones((3, 4))
    V_uniform = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    _, weights_uniform = dot_product_attention(Q_uniform, K_uniform, V_uniform)
    # All keys are equally similar, so weights should be uniform
    expected_weight = 1.0 / 3.0
    assert np.allclose(weights_uniform, expected_weight, atol=1e-5), \
        f"Test 10 failed: expected uniform weights ~{expected_weight}, got {weights_uniform}"

    print("All tests passed!")
