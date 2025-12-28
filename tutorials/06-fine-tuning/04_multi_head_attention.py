# Problem 4: Multi-Head Attention
#
# Implement attention with multiple "heads" that attend in parallel.
#
# Why multiple heads?
# - Each head can learn different attention patterns
# - One head might focus on syntax, another on semantics
# - More expressive than single-head attention
#
# How it works:
# 1. Split Q, K, V into h heads
# 2. Apply attention independently in each head
# 3. Concatenate results
# 4. Project back to original dimension
#
# Dimensions:
#   d_model = total embedding dimension (e.g., 512)
#   h = number of heads (e.g., 8)
#   d_k = d_model / h = per-head dimension (e.g., 64)
#
# Example (d_model=8, h=2, d_k=4):
#   Input: (seq_len, 8)
#   Split into 2 heads: (seq_len, 2, 4)
#   Attention per head: (seq_len, 2, 4)
#   Concat: (seq_len, 8)
#
# ML Relevance: Every transformer uses multi-head attention. GPT-3 has 96 heads!
# Understanding this helps you tune the number of heads and debug attention.

import numpy as np


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Split the last dimension into multiple heads.

    Args:
        x: Input tensor, shape (..., seq_len, d_model)
        num_heads: Number of attention heads

    Returns:
        Reshaped tensor, shape (..., num_heads, seq_len, d_k)
        where d_k = d_model / num_heads

    Example:
        x.shape = (4, 8)  # seq_len=4, d_model=8
        split_heads(x, 2).shape = (2, 4, 4)  # 2 heads, seq_len=4, d_k=4
    """
    # Your solution here
    pass


def concat_heads(x: np.ndarray) -> np.ndarray:
    """
    Reverse of split_heads - concatenate attention heads.

    Args:
        x: Tensor with shape (..., num_heads, seq_len, d_k)

    Returns:
        Tensor with shape (..., seq_len, d_model)
        where d_model = num_heads * d_k

    Example:
        x.shape = (2, 4, 4)  # 2 heads, seq_len=4, d_k=4
        concat_heads(x).shape = (4, 8)  # seq_len=4, d_model=8
    """
    # Your solution here
    pass


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention (reused from previous problems).

    Args:
        query: (..., seq_len_q, d_k)
        key: (..., seq_len_k, d_k)
        value: (..., seq_len_k, d_v)
        mask: Optional mask

    Returns:
        output, weights
    """
    # Your solution here (or import from previous problem)
    pass


def multi_head_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    num_heads: int,
    mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-head attention mechanism.

    Args:
        query: Shape (seq_len_q, d_model)
        key: Shape (seq_len_k, d_model)
        value: Shape (seq_len_k, d_model)
        W_q: Query projection, shape (d_model, d_model)
        W_k: Key projection, shape (d_model, d_model)
        W_v: Value projection, shape (d_model, d_model)
        W_o: Output projection, shape (d_model, d_model)
        num_heads: Number of attention heads
        mask: Optional mask, shape (seq_len_q, seq_len_k)

    Returns:
        Tuple of:
        - output: Shape (seq_len_q, d_model)
        - weights: Shape (num_heads, seq_len_q, seq_len_k)

    Steps:
    1. Project Q, K, V
    2. Split into heads
    3. Apply attention per head
    4. Concat heads
    5. Project output
    """
    # Your solution here
    pass


def analyze_head_attention(
    weights: np.ndarray,
    tokens: list[str]
) -> dict:
    """
    Analyze what each attention head focuses on.

    Args:
        weights: Shape (num_heads, seq_len, seq_len)
        tokens: List of token strings

    Returns:
        Dict with analysis per head:
        - 'head_{i}': {
            'max_attention_pairs': [(query_token, key_token, weight), ...],
            'entropy': float (higher = more uniform attention),
            'pattern': str ('local', 'global', 'mixed')
          }
    """
    # Your solution here
    pass


def visualize_all_heads(
    weights: np.ndarray,
    tokens: list[str]
) -> str:
    """
    Create a visualization of all attention heads.

    Args:
        weights: Shape (num_heads, seq_len, seq_len)
        tokens: List of token strings

    Returns:
        String visualization showing each head's attention pattern
    """
    # Your solution here
    pass


class MultiHeadAttention:
    """
    Multi-head attention as a reusable class.

    This mirrors PyTorch's nn.MultiheadAttention interface.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head attention.

        Args:
            d_model: Total embedding dimension
            num_heads: Number of attention heads
        """
        # Your solution here
        # Initialize W_q, W_k, W_v, W_o with random values
        pass

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention.

        Args:
            query, key, value: Input tensors
            mask: Optional attention mask

        Returns:
            output, attention_weights
        """
        # Your solution here
        pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: Split heads
    x = np.random.randn(4, 8)  # seq_len=4, d_model=8
    split = split_heads(x, num_heads=2)
    assert split.shape == (2, 4, 4), f"Test 1a failed: expected (2, 4, 4), got {split.shape}"

    # Verify the split preserves values
    assert np.allclose(split[0, :, :], x[:, :4]), "Test 1b failed: first head should be first half"
    assert np.allclose(split[1, :, :], x[:, 4:]), "Test 1c failed: second head should be second half"

    # Test 2: Concat heads (inverse of split)
    concat = concat_heads(split)
    assert concat.shape == (4, 8), f"Test 2a failed: expected (4, 8), got {concat.shape}"
    assert np.allclose(concat, x), "Test 2b failed: concat should reverse split"

    # Test 3: Split and concat with different head counts
    x = np.random.randn(6, 12)
    for num_heads in [1, 2, 3, 4, 6, 12]:
        split = split_heads(x, num_heads)
        concat = concat_heads(split)
        assert np.allclose(concat, x), f"Test 3 failed: roundtrip failed for {num_heads} heads"

    # Test 4: Multi-head attention shapes
    seq_len, d_model, num_heads = 5, 16, 4
    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)
    V = np.random.randn(seq_len, d_model)
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    W_o = np.random.randn(d_model, d_model)

    output, weights = multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads)
    assert output.shape == (seq_len, d_model), f"Test 4a failed: expected ({seq_len}, {d_model}), got {output.shape}"
    assert weights.shape == (num_heads, seq_len, seq_len), \
        f"Test 4b failed: expected ({num_heads}, {seq_len}, {seq_len}), got {weights.shape}"

    # Test 5: Attention weights sum to 1 per head
    for h in range(num_heads):
        for q in range(seq_len):
            assert abs(weights[h, q, :].sum() - 1.0) < 1e-5, \
                f"Test 5 failed: head {h}, query {q} weights don't sum to 1"

    # Test 6: Multi-head with mask
    mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)  # Causal mask
    output_masked, weights_masked = multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads, mask)

    # Check no future attention
    for h in range(num_heads):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights_masked[h, i, j] < 1e-6, \
                    f"Test 6 failed: head {h} attends from {i} to future {j}"

    # Test 7: Different heads learn different patterns (with enough capacity)
    # At least some heads should have different attention patterns
    head_patterns = [weights[h] for h in range(num_heads)]
    all_same = all(np.allclose(head_patterns[0], hp) for hp in head_patterns[1:])
    # With random weights, patterns should differ
    assert not all_same, "Test 7 failed: heads should have different patterns"

    # Test 8: Analyze heads
    tokens = ["the", "quick", "brown", "fox", "jumps"]
    analysis = analyze_head_attention(weights, tokens)
    assert len(analysis) == num_heads, f"Test 8a failed: expected {num_heads} heads in analysis"
    assert "head_0" in analysis, "Test 8b failed: should have head_0 in analysis"

    # Test 9: MultiHeadAttention class
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    output_class, weights_class = mha.forward(Q, K, V)
    assert output_class.shape == (seq_len, 16), f"Test 9a failed: wrong output shape"
    assert weights_class.shape == (4, seq_len, seq_len), f"Test 9b failed: wrong weights shape"

    # Test 10: Visualization
    viz = visualize_all_heads(weights[:2], tokens)  # Just first 2 heads
    assert isinstance(viz, str), "Test 10a failed: should return string"
    assert len(viz) > 0, "Test 10b failed: visualization should not be empty"

    print("All tests passed!")
    print("\nMulti-head attention allows the model to attend to different aspects")
    print("of the input simultaneously. Each head can learn specialized patterns!")
