# Problem 7: LoRA Layer From Scratch
#
# Implement the LoRA (Low-Rank Adaptation) layer.
#
# The LoRA idea:
# - Original: y = x @ W
# - LoRA:     y = x @ W + x @ B @ A * (alpha / r)
#
# Where:
# - W: Frozen original weights (d_in, d_out)
# - B: Low-rank factor (d_in, r), initialized to zeros
# - A: Low-rank factor (r, d_out), initialized randomly
# - r: LoRA rank (typically 4-64)
# - alpha: Scaling factor (often = r)
#
# Key insights:
# 1. B starts at 0, so LoRA adds nothing initially
# 2. Only A and B are trained (W is frozen)
# 3. After training, can merge: W_new = W + B @ A * (alpha / r)
#
# ML Relevance: This is THE technique for efficient fine-tuning.
# Understanding this lets you implement and customize LoRA.

import numpy as np


class LoRALayer:
    """
    A LoRA layer that wraps a linear transformation.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int = 4,
        alpha: float = None,
        init_scale: float = 0.01
    ):
        """
        Initialize LoRA layer.

        Args:
            d_in: Input dimension
            d_out: Output dimension
            rank: LoRA rank (r)
            alpha: Scaling factor (defaults to rank)
            init_scale: Scale for random initialization of A
        """
        # Your solution here
        # Initialize:
        # - W: Random weights (the "frozen" base)
        # - B: Zeros (so initial LoRA contribution is 0)
        # - A: Random scaled by init_scale
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: x @ W + x @ B @ A * (alpha / r)

        Args:
            x: Input, shape (..., d_in)

        Returns:
            Output, shape (..., d_out)
        """
        # Your solution here
        pass

    def lora_contribution(self, x: np.ndarray) -> np.ndarray:
        """
        Just the LoRA part: x @ B @ A * (alpha / r)

        Useful for analyzing what LoRA learned.
        """
        # Your solution here
        pass

    def merge_weights(self) -> np.ndarray:
        """
        Merge LoRA into base weights: W_merged = W + B @ A * (alpha / r)

        Returns the merged weight matrix.
        After merging, you can use W_merged directly without LoRA overhead.
        """
        # Your solution here
        pass

    @property
    def num_trainable_params(self) -> int:
        """Number of trainable parameters (just A and B)."""
        # Your solution here
        pass

    @property
    def num_frozen_params(self) -> int:
        """Number of frozen parameters (W)."""
        # Your solution here
        pass

    @property
    def compression_ratio(self) -> float:
        """Ratio of frozen params to trainable params."""
        # Your solution here
        pass

    def get_gradients(self, x: np.ndarray, grad_output: np.ndarray) -> dict:
        """
        Compute gradients for A and B given upstream gradient.

        Args:
            x: Input that was used in forward pass
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Dict with 'grad_A' and 'grad_B'

        This is for understanding - in practice PyTorch autograd handles this.
        """
        # Your solution here
        pass


def compare_full_vs_lora(
    d_in: int,
    d_out: int,
    rank: int
) -> dict:
    """
    Compare full fine-tuning vs LoRA.

    Returns:
    - 'full_params': Parameters for full fine-tuning
    - 'lora_params': Parameters for LoRA
    - 'reduction': How many x fewer params
    - 'memory_full_mb': Approximate memory for full (float32)
    - 'memory_lora_mb': Approximate memory for LoRA
    """
    # Your solution here
    pass


def simulate_training_step(
    lora: LoRALayer,
    x: np.ndarray,
    target: np.ndarray,
    lr: float = 0.01
) -> dict:
    """
    Simulate one training step with gradient descent.

    Args:
        lora: LoRA layer to train
        x: Input
        target: Target output
        lr: Learning rate

    Returns:
        Dict with 'loss_before', 'loss_after', 'A_change', 'B_change'
    """
    # Your solution here
    pass


def visualize_lora_weights(lora: LoRALayer) -> str:
    """
    Visualize the LoRA factors.

    Show:
    - Shape of B and A
    - Norm of each
    - Resulting LoRA matrix norm
    - Comparison to base W norm
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: Initialization
    lora = LoRALayer(d_in=64, d_out=32, rank=4)
    assert hasattr(lora, 'W'), "Test 1a failed: should have W"
    assert hasattr(lora, 'A'), "Test 1b failed: should have A"
    assert hasattr(lora, 'B'), "Test 1c failed: should have B"

    # Test 2: Shapes
    assert lora.W.shape == (64, 32), f"Test 2a failed: W shape"
    assert lora.B.shape == (64, 4), f"Test 2b failed: B shape"
    assert lora.A.shape == (4, 32), f"Test 2c failed: A shape"

    # Test 3: B initialized to zeros
    assert np.allclose(lora.B, 0), "Test 3 failed: B should be zeros initially"

    # Test 4: Forward pass shape
    x = np.random.randn(10, 64)
    output = lora.forward(x)
    assert output.shape == (10, 32), f"Test 4 failed: expected (10, 32), got {output.shape}"

    # Test 5: Initial LoRA contribution is zero (because B=0)
    lora_contrib = lora.lora_contribution(x)
    assert np.allclose(lora_contrib, 0), "Test 5 failed: initial LoRA contrib should be 0"

    # Test 6: Initial output equals base
    base_output = x @ lora.W
    assert np.allclose(output, base_output), "Test 6 failed: initial output should match base"

    # Test 7: After modifying B, LoRA contributes
    lora.B = np.random.randn(64, 4) * 0.1
    output_with_lora = lora.forward(x)
    lora_contrib = lora.lora_contribution(x)
    assert not np.allclose(lora_contrib, 0), "Test 7a failed: LoRA should contribute"
    assert np.allclose(output_with_lora, base_output + lora_contrib), \
        "Test 7b failed: output should be base + LoRA contrib"

    # Test 8: Merge weights
    merged = lora.merge_weights()
    assert merged.shape == (64, 32), f"Test 8a failed: merged shape"
    output_merged = x @ merged
    assert np.allclose(output_with_lora, output_merged), \
        "Test 8b failed: merged output should match LoRA output"

    # Test 9: Trainable params
    assert lora.num_trainable_params == 64 * 4 + 4 * 32, \
        f"Test 9a failed: wrong trainable count"
    assert lora.num_frozen_params == 64 * 32, \
        f"Test 9b failed: wrong frozen count"

    # Test 10: Compression ratio
    expected_compression = (64 * 32) / (64 * 4 + 4 * 32)
    assert abs(lora.compression_ratio - expected_compression) < 0.01, \
        f"Test 10 failed: expected {expected_compression}, got {lora.compression_ratio}"

    # Test 11: Different ranks
    for rank in [1, 8, 16, 32]:
        lora_r = LoRALayer(d_in=128, d_out=64, rank=rank)
        assert lora_r.A.shape == (rank, 64), f"Test 11a failed: rank={rank}"
        assert lora_r.B.shape == (128, rank), f"Test 11b failed: rank={rank}"

    # Test 12: Alpha scaling
    lora_alpha = LoRALayer(d_in=32, d_out=32, rank=4, alpha=8)
    lora_alpha.B = np.ones((32, 4))
    lora_alpha.A = np.ones((4, 32))
    x_test = np.ones((1, 32))
    contrib = lora_alpha.lora_contribution(x_test)
    # Contribution should be scaled by alpha/r = 8/4 = 2
    expected_scale = 8 / 4
    assert np.allclose(contrib, 32 * 4 * expected_scale), \
        f"Test 12 failed: alpha scaling wrong"

    # Test 13: Compare full vs LoRA
    comparison = compare_full_vs_lora(4096, 4096, rank=8)
    assert comparison['lora_params'] < comparison['full_params'], \
        "Test 13 failed: LoRA should have fewer params"
    assert comparison['reduction'] > 100, \
        "Test 13b failed: should have >100x reduction"

    # Test 14: Gradients
    lora2 = LoRALayer(d_in=16, d_out=8, rank=2)
    lora2.B = np.random.randn(16, 2) * 0.1
    x_grad = np.random.randn(5, 16)
    grad_out = np.random.randn(5, 8)
    grads = lora2.get_gradients(x_grad, grad_out)
    assert 'grad_A' in grads, "Test 14a failed: missing grad_A"
    assert 'grad_B' in grads, "Test 14b failed: missing grad_B"
    assert grads['grad_A'].shape == (2, 8), f"Test 14c failed: grad_A shape"
    assert grads['grad_B'].shape == (16, 2), f"Test 14d failed: grad_B shape"

    # Test 15: Visualization
    viz = visualize_lora_weights(lora)
    assert isinstance(viz, str), "Test 15 failed: should return string"

    print("All tests passed!")
    print("\nYou've implemented LoRA from scratch!")
    print("Key insight: By training just B and A, we get huge parameter savings")
    print("while still adapting the model effectively.")
