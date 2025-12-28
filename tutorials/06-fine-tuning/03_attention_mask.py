# Problem 3: Causal Attention Mask
#
# Implement masking to prevent attending to future tokens.
# This is essential for autoregressive models like GPT.
#
# Why we need masking:
# - In training, we process whole sequences at once
# - But each position should only see previous positions
# - Without masking, the model "cheats" by looking ahead
#
# How masking works:
# 1. Create a mask matrix where future positions are marked
# 2. Add -infinity to masked positions before softmax
# 3. Softmax turns -inf into 0, effectively ignoring those positions
#
# Example for sequence length 4:
#   Mask = [[0, -inf, -inf, -inf],   # Position 0 sees only itself
#           [0,    0, -inf, -inf],   # Position 1 sees 0, 1
#           [0,    0,    0, -inf],   # Position 2 sees 0, 1, 2
#           [0,    0,    0,    0]]   # Position 3 sees all
#
# ML Relevance: This is how GPT, Llama, and all decoder-only models work.
# Understanding masking is crucial for debugging generation issues.

import numpy as np


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (lower triangular) attention mask.

    Args:
        seq_len: Length of the sequence

    Returns:
        Mask of shape (seq_len, seq_len)
        - 0 for positions that CAN be attended to
        - -infinity for positions that CANNOT be attended to

    Example (seq_len=3):
        [[0, -inf, -inf],
         [0,    0, -inf],
         [0,    0,    0]]
    """
    # Your solution here
    pass


def create_padding_mask(lengths: list[int], max_len: int) -> np.ndarray:
    """
    Create a padding mask for variable-length sequences.

    Args:
        lengths: List of actual sequence lengths
        max_len: Maximum (padded) sequence length

    Returns:
        Mask of shape (batch_size, max_len)
        - 0 for real tokens
        - -infinity for padding tokens

    Example:
        lengths = [2, 3], max_len = 4
        -> [[0, 0, -inf, -inf],
            [0, 0,    0, -inf]]
    """
    # Your solution here
    pass


def combine_masks(
    causal_mask: np.ndarray,
    padding_mask: np.ndarray
) -> np.ndarray:
    """
    Combine causal and padding masks.

    Args:
        causal_mask: Shape (seq_len, seq_len)
        padding_mask: Shape (batch_size, seq_len)

    Returns:
        Combined mask of shape (batch_size, seq_len, seq_len)

    A position is masked if EITHER:
    - It's in the future (causal)
    - It's a padding token
    """
    # Your solution here
    pass


def apply_mask(
    scores: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Apply mask to attention scores before softmax.

    Args:
        scores: Attention scores, shape (..., seq_len_q, seq_len_k)
        mask: Mask with same shape as scores (broadcastable)

    Returns:
        Masked scores (add mask to scores)
    """
    # Your solution here
    pass


def masked_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention with optional mask.

    Args:
        query: Shape (seq_len_q, d_k)
        key: Shape (seq_len_k, d_k)
        value: Shape (seq_len_k, d_v)
        mask: Optional mask, shape (seq_len_q, seq_len_k)

    Returns:
        Tuple of (output, weights)
    """
    # Your solution here
    pass


def causal_self_attention(
    x: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full causal self-attention with projections.

    Args:
        x: Input sequence, shape (seq_len, d_model)
        W_q: Query projection, shape (d_model, d_k)
        W_k: Key projection, shape (d_model, d_k)
        W_v: Value projection, shape (d_model, d_v)

    Returns:
        Tuple of (output, weights)

    Steps:
    1. Project inputs to Q, K, V
    2. Create causal mask
    3. Compute masked attention
    """
    # Your solution here
    pass


def visualize_causal_mask(seq_len: int, tokens: list[str] = None) -> str:
    """
    Visualize the causal mask pattern.

    Args:
        seq_len: Sequence length
        tokens: Optional token labels

    Returns:
        String visualization showing which positions can attend to which

    Example:
        visualize_causal_mask(3, ["a", "b", "c"])
        ->
              | a | b | c |
        ------+---+---+---+
        a     | 1 | 0 | 0 |
        b     | 1 | 1 | 0 |
        c     | 1 | 1 | 1 |
    """
    # Your solution here
    pass


def verify_no_future_leakage(weights: np.ndarray) -> bool:
    """
    Verify that attention weights don't attend to future positions.

    Args:
        weights: Attention weights, shape (seq_len, seq_len)

    Returns:
        True if no position attends to future positions
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Create causal mask
    mask = create_causal_mask(4)
    assert mask.shape == (4, 4), f"Test 1a failed: expected (4, 4), got {mask.shape}"
    assert mask[0, 1] == -np.inf, "Test 1b failed: position 0 should not see position 1"
    assert mask[2, 1] == 0, "Test 1c failed: position 2 should see position 1"
    assert mask[3, 0] == 0, "Test 1d failed: position 3 should see position 0"

    # Test 2: Causal mask is lower triangular
    mask = create_causal_mask(5)
    for i in range(5):
        for j in range(5):
            if j <= i:
                assert mask[i, j] == 0, f"Test 2a failed: [{i},{j}] should be 0"
            else:
                assert mask[i, j] == -np.inf, f"Test 2b failed: [{i},{j}] should be -inf"

    # Test 3: Padding mask
    padding = create_padding_mask([2, 4], max_len=5)
    assert padding.shape == (2, 5), f"Test 3a failed: expected (2, 5), got {padding.shape}"
    assert padding[0, 1] == 0, "Test 3b failed: position 1 in seq 0 is real"
    assert padding[0, 2] == -np.inf, "Test 3c failed: position 2 in seq 0 is padding"
    assert padding[1, 3] == 0, "Test 3d failed: position 3 in seq 1 is real"

    # Test 4: Apply mask
    scores = np.ones((3, 3))
    mask = create_causal_mask(3)
    masked_scores = apply_mask(scores, mask)
    assert masked_scores[0, 1] == -np.inf, "Test 4a failed: masked position should be -inf"
    assert masked_scores[2, 0] == 1, "Test 4b failed: unmasked position should be unchanged"

    # Test 5: Masked attention
    seq_len, d = 4, 8
    Q = np.random.randn(seq_len, d)
    K = np.random.randn(seq_len, d)
    V = np.random.randn(seq_len, d)
    mask = create_causal_mask(seq_len)

    output, weights = masked_attention(Q, K, V, mask)
    assert output.shape == (seq_len, d), f"Test 5a failed: expected ({seq_len}, {d}), got {output.shape}"

    # Test 6: No future leakage
    assert verify_no_future_leakage(weights), "Test 6 failed: attention should not leak to future"

    # Test 7: First position only attends to itself
    assert abs(weights[0, 0] - 1.0) < 1e-5, "Test 7 failed: first position should fully attend to itself"

    # Test 8: Combine masks
    causal = create_causal_mask(4)
    padding = create_padding_mask([2, 3], max_len=4)
    combined = combine_masks(causal, padding)
    assert combined.shape == (2, 4, 4), f"Test 8a failed: expected (2, 4, 4), got {combined.shape}"
    # Position 0, sequence 0: can only see position 0 (itself)
    assert combined[0, 0, 0] == 0, "Test 8b failed: should be able to see itself"
    # Position 1 in sequence 0 should not see position 2 (causal) and position 2 is also padding
    assert combined[0, 1, 2] == -np.inf, "Test 8c failed: should be masked"

    # Test 9: Causal self-attention
    d_model, d_k = 16, 8
    x = np.random.randn(5, d_model)
    W_q = np.random.randn(d_model, d_k)
    W_k = np.random.randn(d_model, d_k)
    W_v = np.random.randn(d_model, d_k)

    output, weights = causal_self_attention(x, W_q, W_k, W_v)
    assert output.shape == (5, d_k), f"Test 9a failed: expected (5, {d_k}), got {output.shape}"
    assert verify_no_future_leakage(weights), "Test 9b failed: causal self-attention should not leak"

    # Test 10: Visualization
    viz = visualize_causal_mask(3, ["a", "b", "c"])
    assert isinstance(viz, str), "Test 10a failed: should return string"
    assert "a" in viz, "Test 10b failed: should contain token labels"

    print("All tests passed!")
    print("\nCausal masking ensures the model can only see past tokens.")
    print("This is what makes GPT/Llama autoregressive - they generate one token at a time.")
