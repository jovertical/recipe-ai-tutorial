# Problem 5: Q, K, V Projections
#
# Understand the linear projections that create Q, K, V from input embeddings.
# These are the weight matrices that LoRA will modify.
#
# In a transformer:
#   Q = X @ W_q
#   K = X @ W_k
#   V = X @ W_v
#   Output = Attention(Q, K, V) @ W_o
#
# Where:
#   X: Input embeddings (seq_len, d_model)
#   W_q, W_k, W_v: Projection weights (d_model, d_k or d_model, d_v)
#   W_o: Output projection (d_v, d_model)
#
# These 4 weight matrices (W_q, W_k, W_v, W_o) are often the targets for LoRA
# because they have the most impact on attention behavior.
#
# ML Relevance: Understanding what these projections do helps you decide
# which layers to apply LoRA to. Modifying W_q/W_k changes what the model
# attends to; modifying W_v/W_o changes what information is extracted.

import numpy as np


class QKVProjection:
    """
    Linear projections for Query, Key, and Value.
    """

    def __init__(self, d_model: int, d_k: int, d_v: int = None):
        """
        Initialize Q, K, V projections.

        Args:
            d_model: Input embedding dimension
            d_k: Query and Key dimension
            d_v: Value dimension (defaults to d_k if not specified)
        """
        # Your solution here
        # Initialize W_q, W_k, W_v with random values
        # Use Xavier initialization: scale = sqrt(2 / (d_in + d_out))
        pass

    def project_query(self, x: np.ndarray) -> np.ndarray:
        """Project input to query space."""
        # Your solution here
        pass

    def project_key(self, x: np.ndarray) -> np.ndarray:
        """Project input to key space."""
        # Your solution here
        pass

    def project_value(self, x: np.ndarray) -> np.ndarray:
        """Project input to value space."""
        # Your solution here
        pass

    def project_all(self, x: np.ndarray) -> tuple:
        """
        Project input to Q, K, V simultaneously.

        Returns (Q, K, V) tuple
        """
        # Your solution here
        pass

    @property
    def num_parameters(self) -> int:
        """Total number of parameters in all projections."""
        # Your solution here
        pass


class OutputProjection:
    """
    Output projection that maps attention output back to model dimension.
    """

    def __init__(self, d_v: int, d_model: int):
        """
        Initialize output projection.

        Args:
            d_v: Value dimension (attention output dimension)
            d_model: Model dimension (output dimension)
        """
        # Your solution here
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply output projection."""
        # Your solution here
        pass

    @property
    def num_parameters(self) -> int:
        """Number of parameters."""
        # Your solution here
        pass


def full_attention_block(
    x: np.ndarray,
    qkv_proj: QKVProjection,
    output_proj: OutputProjection,
    mask: np.ndarray = None
) -> np.ndarray:
    """
    Complete attention block with all projections.

    Args:
        x: Input, shape (seq_len, d_model)
        qkv_proj: Q, K, V projection module
        output_proj: Output projection module
        mask: Optional attention mask

    Returns:
        Output, shape (seq_len, d_model)

    This is the full self-attention block in a transformer layer.
    """
    # Your solution here
    # 1. Project to Q, K, V
    # 2. Compute scaled dot-product attention
    # 3. Project output
    pass


def analyze_projection_impact(
    x: np.ndarray,
    qkv_proj: QKVProjection
) -> dict:
    """
    Analyze how projections transform the input.

    Returns:
    - 'q_norm_change': How much Q changes input norm
    - 'k_norm_change': How much K changes input norm
    - 'v_norm_change': How much V changes input norm
    - 'q_rank': Estimated rank of Q projection
    - 'cosine_q_k': Cosine similarity between Q and K outputs
    """
    # Your solution here
    pass


def projection_parameter_count(d_model: int, d_k: int, d_v: int) -> dict:
    """
    Calculate parameter counts for attention projections.

    Returns dict with counts for:
    - 'W_q': Query projection params
    - 'W_k': Key projection params
    - 'W_v': Value projection params
    - 'W_o': Output projection params
    - 'total': Total params
    - 'total_with_bias': If we added bias terms
    """
    # Your solution here
    pass


def compare_projection_strategies(d_model: int) -> dict:
    """
    Compare different projection dimension choices.

    Common strategies:
    - d_k = d_v = d_model (full dimension)
    - d_k = d_v = d_model / num_heads (typical for MHA)
    - d_k < d_v (more capacity for values)

    Returns comparison of parameter counts and expressiveness.
    """
    # Your solution here
    pass


def visualize_projection_weights(qkv_proj: QKVProjection) -> str:
    """
    Create a visualization of projection weight matrices.

    Show:
    - Weight matrix dimensions
    - Value distribution (mean, std)
    - Sparsity pattern (if any)
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: QKV Projection initialization
    d_model, d_k, d_v = 64, 32, 32
    qkv = QKVProjection(d_model, d_k, d_v)
    assert hasattr(qkv, 'W_q'), "Test 1a failed: should have W_q"
    assert hasattr(qkv, 'W_k'), "Test 1b failed: should have W_k"
    assert hasattr(qkv, 'W_v'), "Test 1c failed: should have W_v"

    # Test 2: Projection shapes
    seq_len = 10
    x = np.random.randn(seq_len, d_model)

    Q = qkv.project_query(x)
    K = qkv.project_key(x)
    V = qkv.project_value(x)

    assert Q.shape == (seq_len, d_k), f"Test 2a failed: expected ({seq_len}, {d_k}), got {Q.shape}"
    assert K.shape == (seq_len, d_k), f"Test 2b failed: expected ({seq_len}, {d_k}), got {K.shape}"
    assert V.shape == (seq_len, d_v), f"Test 2c failed: expected ({seq_len}, {d_v}), got {V.shape}"

    # Test 3: Project all
    Q2, K2, V2 = qkv.project_all(x)
    assert np.allclose(Q, Q2), "Test 3a failed: project_all Q should match project_query"
    assert np.allclose(K, K2), "Test 3b failed: project_all K should match project_key"
    assert np.allclose(V, V2), "Test 3c failed: project_all V should match project_value"

    # Test 4: Parameter count
    expected_params = d_model * d_k + d_model * d_k + d_model * d_v
    assert qkv.num_parameters == expected_params, \
        f"Test 4 failed: expected {expected_params}, got {qkv.num_parameters}"

    # Test 5: Output projection
    output_proj = OutputProjection(d_v, d_model)
    attn_out = np.random.randn(seq_len, d_v)
    final_out = output_proj.forward(attn_out)
    assert final_out.shape == (seq_len, d_model), \
        f"Test 5 failed: expected ({seq_len}, {d_model}), got {final_out.shape}"

    # Test 6: Full attention block
    output = full_attention_block(x, qkv, output_proj)
    assert output.shape == (seq_len, d_model), \
        f"Test 6a failed: expected ({seq_len}, {d_model}), got {output.shape}"

    # Test 7: Full attention block with mask
    mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
    output_masked = full_attention_block(x, qkv, output_proj, mask)
    assert output_masked.shape == (seq_len, d_model), "Test 7 failed: masked output wrong shape"

    # Test 8: Analyze projection impact
    analysis = analyze_projection_impact(x, qkv)
    assert 'q_norm_change' in analysis, "Test 8a failed: missing q_norm_change"
    assert 'cosine_q_k' in analysis, "Test 8b failed: missing cosine_q_k"

    # Test 9: Parameter count calculation
    counts = projection_parameter_count(512, 64, 64)
    assert counts['W_q'] == 512 * 64, f"Test 9a failed: wrong W_q count"
    assert counts['W_o'] == 64 * 512, f"Test 9b failed: wrong W_o count"
    assert counts['total'] == 4 * 512 * 64, f"Test 9c failed: wrong total"

    # Test 10: Compare strategies
    comparison = compare_projection_strategies(512)
    assert isinstance(comparison, dict), "Test 10 failed: should return dict"

    # Test 11: Different d_v
    qkv_diff = QKVProjection(d_model=64, d_k=32, d_v=48)
    x_test = np.random.randn(5, 64)
    Q, K, V = qkv_diff.project_all(x_test)
    assert V.shape == (5, 48), f"Test 11 failed: expected V shape (5, 48), got {V.shape}"

    # Test 12: Visualization
    viz = visualize_projection_weights(qkv)
    assert isinstance(viz, str), "Test 12a failed: should return string"
    assert len(viz) > 0, "Test 12b failed: visualization should not be empty"

    print("All tests passed!")
    print("\nQ, K, V projections are the weights that LoRA will modify.")
    print("They transform input embeddings into the attention space.")
