# Problem 8: LoRA Injection
#
# Apply LoRA to existing linear layers in a model.
# This is how PEFT/LoRA libraries work under the hood.
#
# The process:
# 1. Identify target layers (typically attention Q, K, V, O projections)
# 2. Wrap each target layer with LoRA
# 3. Freeze original weights
# 4. Train only LoRA parameters
#
# Example:
#   Original: nn.Linear(768, 768)  # 589,824 params
#   With LoRA (r=8): Same layer + B(768, 8) + A(8, 768)  # 12,288 trainable
#
# ML Relevance: This is exactly what happens when you use PEFT's get_peft_model().
# Understanding this helps you customize which layers to target.

import numpy as np


class LinearLayer:
    """Simple linear layer (mimics nn.Linear)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        # Xavier initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features) if bias else None
        self.requires_grad = True  # Trainable by default

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    @property
    def num_parameters(self) -> int:
        count = self.weight.size
        if self.bias is not None:
            count += self.bias.size
        return count


class LoRALinear:
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        base_layer: LinearLayer,
        rank: int = 4,
        alpha: float = None,
        dropout: float = 0.0
    ):
        """
        Wrap a linear layer with LoRA.

        Args:
            base_layer: The original linear layer to wrap
            rank: LoRA rank
            alpha: Scaling factor (defaults to rank)
            dropout: Dropout rate for LoRA path (not implemented, for API compat)
        """
        # Your solution here
        # - Store base layer
        # - Initialize A (random) and B (zeros)
        # - Freeze base layer weights
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: base_layer(x) + x @ B @ A * scaling
        """
        # Your solution here
        pass

    def merge_and_unload(self) -> LinearLayer:
        """
        Merge LoRA into base weights and return a regular LinearLayer.

        After this, you can use the merged layer without LoRA overhead.
        """
        # Your solution here
        pass

    @property
    def trainable_parameters(self) -> dict:
        """Return dict of trainable parameters (A and B)."""
        # Your solution here
        pass

    @property
    def num_trainable(self) -> int:
        """Count of trainable parameters."""
        # Your solution here
        pass


def apply_lora_to_linear(
    layer: LinearLayer,
    rank: int = 4,
    alpha: float = None
) -> LoRALinear:
    """
    Convert a LinearLayer to LoRALinear.

    This is a convenience function that wraps the layer.
    """
    # Your solution here
    pass


def freeze_base_model(layers: list[LoRALinear]) -> None:
    """
    Ensure all base layers are frozen (requires_grad = False).

    In PyTorch, this would set param.requires_grad = False
    Here we just verify/set the flag.
    """
    # Your solution here
    pass


def count_trainable_params(layers: list) -> dict:
    """
    Count trainable vs frozen parameters across layers.

    Returns:
    - 'trainable': total trainable params
    - 'frozen': total frozen params
    - 'total': all params
    - 'trainable_percent': % of params that are trainable
    """
    # Your solution here
    pass


def get_lora_state_dict(layers: list[LoRALinear]) -> dict:
    """
    Get only the LoRA parameters (A and B matrices).

    This is what you save - much smaller than full model!

    Returns dict like:
    {
        'layer_0.lora_A': array,
        'layer_0.lora_B': array,
        'layer_1.lora_A': array,
        ...
    }
    """
    # Your solution here
    pass


def load_lora_state_dict(layers: list[LoRALinear], state_dict: dict) -> None:
    """
    Load LoRA parameters from a state dict.
    """
    # Your solution here
    pass


class SimpleTransformerBlock:
    """A simplified transformer block for testing LoRA injection."""

    def __init__(self, d_model: int = 256, num_heads: int = 4):
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # Attention projections (targets for LoRA)
        self.q_proj = LinearLayer(d_model, d_model)
        self.k_proj = LinearLayer(d_model, d_model)
        self.v_proj = LinearLayer(d_model, d_model)
        self.o_proj = LinearLayer(d_model, d_model)

        # FFN (could also be LoRA targets)
        self.ffn_up = LinearLayer(d_model, d_model * 4)
        self.ffn_down = LinearLayer(d_model * 4, d_model)

    def get_attention_layers(self) -> list[LinearLayer]:
        """Get layers typically targeted by LoRA."""
        return [self.q_proj, self.k_proj, self.v_proj, self.o_proj]

    def get_all_layers(self) -> list[LinearLayer]:
        """Get all linear layers."""
        return [
            self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            self.ffn_up, self.ffn_down
        ]


def inject_lora_into_block(
    block: SimpleTransformerBlock,
    rank: int = 4,
    target_modules: list[str] = None
) -> dict:
    """
    Inject LoRA into a transformer block.

    Args:
        block: The transformer block
        rank: LoRA rank
        target_modules: Which modules to target ('q', 'k', 'v', 'o', 'ffn_up', 'ffn_down')
                       Defaults to ['q', 'k', 'v', 'o']

    Returns:
        Dict mapping module names to LoRALinear layers
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: Basic LinearLayer
    linear = LinearLayer(64, 32)
    x = np.random.randn(10, 64)
    output = linear.forward(x)
    assert output.shape == (10, 32), f"Test 1 failed: expected (10, 32), got {output.shape}"

    # Test 2: Wrap with LoRA
    lora_linear = LoRALinear(linear, rank=4)
    output_lora = lora_linear.forward(x)
    assert output_lora.shape == (10, 32), f"Test 2 failed: wrong output shape"

    # Test 3: Initial output matches base (B=0)
    assert np.allclose(output, output_lora), "Test 3 failed: initial LoRA should match base"

    # Test 4: Trainable parameters
    trainable = lora_linear.trainable_parameters
    assert 'A' in trainable, "Test 4a failed: should have A"
    assert 'B' in trainable, "Test 4b failed: should have B"
    assert trainable['A'].shape == (4, 32), f"Test 4c failed: A shape"
    assert trainable['B'].shape == (64, 4), f"Test 4d failed: B shape"

    # Test 5: Trainable count
    expected_trainable = 64 * 4 + 4 * 32
    assert lora_linear.num_trainable == expected_trainable, \
        f"Test 5 failed: expected {expected_trainable}, got {lora_linear.num_trainable}"

    # Test 6: apply_lora_to_linear helper
    linear2 = LinearLayer(128, 64)
    lora2 = apply_lora_to_linear(linear2, rank=8)
    assert isinstance(lora2, LoRALinear), "Test 6 failed: should return LoRALinear"

    # Test 7: Count trainable params
    layers = [lora_linear, lora2]
    counts = count_trainable_params(layers)
    assert 'trainable' in counts, "Test 7a failed: missing trainable"
    assert 'frozen' in counts, "Test 7b failed: missing frozen"
    assert counts['trainable'] < counts['frozen'], "Test 7c failed: LoRA should have fewer trainable"

    # Test 8: Merge and unload
    lora_linear.trainable_parameters['B'][:] = np.random.randn(64, 4) * 0.1
    merged = lora_linear.merge_and_unload()
    assert isinstance(merged, LinearLayer), "Test 8a failed: should return LinearLayer"
    # Verify merged output matches LoRA output
    output_merged = merged.forward(x)
    output_lora_after = lora_linear.forward(x)
    assert np.allclose(output_merged, output_lora_after), \
        "Test 8b failed: merged should match LoRA output"

    # Test 9: State dict
    linear3 = LinearLayer(32, 32)
    lora3 = apply_lora_to_linear(linear3, rank=4)
    lora3.trainable_parameters['A'][:] = np.ones((4, 32))
    lora3.trainable_parameters['B'][:] = np.ones((32, 4))

    state = get_lora_state_dict([lora3])
    assert 'layer_0.lora_A' in state, "Test 9a failed: missing A in state"
    assert 'layer_0.lora_B' in state, "Test 9b failed: missing B in state"

    # Test 10: Load state dict
    linear4 = LinearLayer(32, 32)
    lora4 = apply_lora_to_linear(linear4, rank=4)
    load_lora_state_dict([lora4], state)
    assert np.allclose(lora4.trainable_parameters['A'], np.ones((4, 32))), \
        "Test 10 failed: A not loaded correctly"

    # Test 11: SimpleTransformerBlock
    block = SimpleTransformerBlock(d_model=128, num_heads=4)
    attn_layers = block.get_attention_layers()
    assert len(attn_layers) == 4, f"Test 11 failed: expected 4 attention layers"

    # Test 12: Inject LoRA into block
    lora_layers = inject_lora_into_block(block, rank=8)
    assert 'q' in lora_layers, "Test 12a failed: q should be wrapped"
    assert 'v' in lora_layers, "Test 12b failed: v should be wrapped"
    assert isinstance(lora_layers['q'], LoRALinear), "Test 12c failed: should be LoRALinear"

    # Test 13: Custom target modules
    block2 = SimpleTransformerBlock(d_model=64)
    lora_custom = inject_lora_into_block(block2, rank=4, target_modules=['q', 'v'])
    assert 'q' in lora_custom, "Test 13a failed: q should be wrapped"
    assert 'k' not in lora_custom, "Test 13b failed: k should not be wrapped"

    # Test 14: Freeze check
    freeze_base_model(list(lora_layers.values()))
    for name, layer in lora_layers.items():
        assert not layer.base_layer.requires_grad, f"Test 14 failed: {name} base should be frozen"

    print("All tests passed!")
    print("\nYou've learned how to inject LoRA into existing models!")
    print("This is exactly what PEFT does under the hood.")
