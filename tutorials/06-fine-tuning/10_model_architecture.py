# Problem 10: Model Architecture Inspection
#
# Before applying LoRA, you need to understand your model's structure.
# This exercise teaches you to inspect and navigate model layers.
#
# Key things to identify:
# 1. Layer names and hierarchy
# 2. Parameter counts per layer
# 3. Which layers are attention vs FFN
# 4. Which layers to target for LoRA
#
# Example model structure (simplified LLaMA):
#   model.embed_tokens: Embedding(32000, 4096)
#   model.layers.0.self_attn.q_proj: Linear(4096, 4096)
#   model.layers.0.self_attn.k_proj: Linear(4096, 4096)
#   model.layers.0.self_attn.v_proj: Linear(4096, 4096)
#   model.layers.0.self_attn.o_proj: Linear(4096, 4096)
#   model.layers.0.mlp.up_proj: Linear(4096, 11008)
#   ...
#
# ML Relevance: You need to know layer names to configure LoRA targets.
# Different models use different naming conventions!

import numpy as np

# We'll use transformers if available, otherwise mock it
try:
    from transformers import AutoModelForCausalLM, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class MockLinear:
    """Mock linear layer for testing without PyTorch."""
    def __init__(self, in_features: int, out_features: int, name: str = ""):
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.weight = np.random.randn(out_features, in_features)
        self.bias = np.zeros(out_features)

    @property
    def num_parameters(self) -> int:
        return self.weight.size + self.bias.size


class MockAttention:
    """Mock attention module."""
    def __init__(self, d_model: int, name_prefix: str = ""):
        self.q_proj = MockLinear(d_model, d_model, f"{name_prefix}.q_proj")
        self.k_proj = MockLinear(d_model, d_model, f"{name_prefix}.k_proj")
        self.v_proj = MockLinear(d_model, d_model, f"{name_prefix}.v_proj")
        self.o_proj = MockLinear(d_model, d_model, f"{name_prefix}.o_proj")


class MockMLP:
    """Mock MLP/FFN module."""
    def __init__(self, d_model: int, d_ff: int, name_prefix: str = ""):
        self.up_proj = MockLinear(d_model, d_ff, f"{name_prefix}.up_proj")
        self.down_proj = MockLinear(d_ff, d_model, f"{name_prefix}.down_proj")
        self.gate_proj = MockLinear(d_model, d_ff, f"{name_prefix}.gate_proj")


class MockTransformerLayer:
    """Mock transformer layer."""
    def __init__(self, d_model: int, d_ff: int, layer_idx: int):
        prefix = f"layers.{layer_idx}"
        self.self_attn = MockAttention(d_model, f"{prefix}.self_attn")
        self.mlp = MockMLP(d_model, d_ff, f"{prefix}.mlp")


class MockLlamaModel:
    """Mock LLaMA-like model for testing."""
    def __init__(self, num_layers: int = 4, d_model: int = 256, d_ff: int = 512, vocab_size: int = 1000):
        self.config = {
            'num_layers': num_layers,
            'd_model': d_model,
            'd_ff': d_ff,
            'vocab_size': vocab_size
        }
        self.embed_tokens = MockLinear(vocab_size, d_model, "embed_tokens")
        self.layers = [MockTransformerLayer(d_model, d_ff, i) for i in range(num_layers)]
        self.lm_head = MockLinear(d_model, vocab_size, "lm_head")


def load_small_model(model_name: str = "mock"):
    """
    Load a small model for inspection.

    Args:
        model_name: Model to load. Use "mock" for testing without GPU.
                   Real options: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    Returns:
        Model object (mock or real)
    """
    # Your solution here
    # If model_name == "mock", return MockLlamaModel
    # Otherwise, try to load with transformers
    pass


def print_layer_names(model) -> list[str]:
    """
    Get all named modules/layers in the model.

    Returns list of layer names like:
    ['embed_tokens', 'layers.0.self_attn.q_proj', ...]
    """
    # Your solution here
    pass


def get_layer_by_name(model, name: str):
    """
    Get a specific layer by its name.

    Example:
        layer = get_layer_by_name(model, "layers.0.self_attn.q_proj")
    """
    # Your solution here
    pass


def model_size_analysis(model) -> dict:
    """
    Analyze model size by component.

    Returns:
    - 'total_params': Total parameter count
    - 'embedding_params': Parameters in embeddings
    - 'attention_params': Parameters in attention layers
    - 'mlp_params': Parameters in MLP/FFN layers
    - 'other_params': Other parameters
    - 'params_by_layer': Dict of layer_name -> param_count
    """
    # Your solution here
    pass


def find_linear_layers(model) -> list[str]:
    """
    Find all linear/dense layers in the model.

    These are the candidates for LoRA.

    Returns list of layer names.
    """
    # Your solution here
    pass


def find_attention_projections(model) -> dict:
    """
    Find attention projection layers (Q, K, V, O).

    Returns dict like:
    {
        'q_proj': ['layers.0.self_attn.q_proj', 'layers.1.self_attn.q_proj', ...],
        'k_proj': [...],
        'v_proj': [...],
        'o_proj': [...]
    }
    """
    # Your solution here
    pass


def find_mlp_layers(model) -> dict:
    """
    Find MLP/FFN layers.

    Returns dict like:
    {
        'up_proj': [...],
        'down_proj': [...],
        'gate_proj': [...]  # For gated models like LLaMA
    }
    """
    # Your solution here
    pass


def recommend_lora_targets(model, budget_ratio: float = 0.01) -> dict:
    """
    Recommend which layers to target with LoRA.

    Args:
        model: The model to analyze
        budget_ratio: Target trainable params as fraction of total

    Returns:
    - 'recommended_targets': List of layer name patterns
    - 'estimated_params': Estimated trainable params
    - 'actual_ratio': Actual ratio achieved
    - 'reasoning': Why these targets were chosen
    """
    # Your solution here
    pass


def compare_model_architectures() -> dict:
    """
    Compare naming conventions across popular models.

    Returns dict showing typical layer names for:
    - LLaMA/Mistral
    - GPT-2
    - BERT
    - T5
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Load mock model
    model = load_small_model("mock")
    assert model is not None, "Test 1 failed: should load model"

    # Test 2: Print layer names
    names = print_layer_names(model)
    assert len(names) > 0, "Test 2a failed: should have layer names"
    assert any("q_proj" in name for name in names), "Test 2b failed: should have q_proj"

    # Test 3: Get layer by name
    layer = get_layer_by_name(model, "layers.0.self_attn.q_proj")
    assert layer is not None, "Test 3a failed: should find layer"
    assert hasattr(layer, 'in_features'), "Test 3b failed: should be linear layer"

    # Test 4: Model size analysis
    analysis = model_size_analysis(model)
    assert 'total_params' in analysis, "Test 4a failed: missing total_params"
    assert 'attention_params' in analysis, "Test 4b failed: missing attention_params"
    assert analysis['total_params'] > 0, "Test 4c failed: should have params"

    # Test 5: Param breakdown adds up
    component_sum = (
        analysis['embedding_params'] +
        analysis['attention_params'] +
        analysis['mlp_params'] +
        analysis['other_params']
    )
    # Allow some tolerance for rounding
    assert abs(component_sum - analysis['total_params']) < analysis['total_params'] * 0.01, \
        "Test 5 failed: component params should sum to total"

    # Test 6: Find linear layers
    linear_layers = find_linear_layers(model)
    assert len(linear_layers) > 0, "Test 6a failed: should find linear layers"
    assert all(isinstance(name, str) for name in linear_layers), "Test 6b failed: should be strings"

    # Test 7: Find attention projections
    attn_projs = find_attention_projections(model)
    assert 'q_proj' in attn_projs, "Test 7a failed: should find q_proj"
    assert 'v_proj' in attn_projs, "Test 7b failed: should find v_proj"
    assert len(attn_projs['q_proj']) == model.config['num_layers'], \
        "Test 7c failed: should have q_proj in each layer"

    # Test 8: Find MLP layers
    mlp_layers = find_mlp_layers(model)
    assert 'up_proj' in mlp_layers, "Test 8a failed: should find up_proj"
    assert 'down_proj' in mlp_layers, "Test 8b failed: should find down_proj"

    # Test 9: Recommend targets
    recommendation = recommend_lora_targets(model)
    assert 'recommended_targets' in recommendation, "Test 9a failed: missing targets"
    assert 'reasoning' in recommendation, "Test 9b failed: missing reasoning"
    assert len(recommendation['recommended_targets']) > 0, "Test 9c failed: should recommend something"

    # Test 10: Model architecture comparison
    architectures = compare_model_architectures()
    assert 'llama' in architectures or 'LLaMA' in str(architectures), \
        "Test 10a failed: should include LLaMA"
    assert 'gpt2' in architectures or 'GPT' in str(architectures), \
        "Test 10b failed: should include GPT"

    # Test 11: Layer names follow pattern
    for name in names:
        parts = name.split('.')
        assert all(isinstance(p, str) for p in parts), \
            f"Test 11 failed: invalid name format: {name}"

    # Test 12: Nested layer access
    attn = get_layer_by_name(model, "layers.0.self_attn")
    assert attn is not None, "Test 12a failed: should get attention module"
    assert hasattr(attn, 'q_proj'), "Test 12b failed: attention should have q_proj"

    print("All tests passed!")
    print("\nModel inspection is crucial for LoRA configuration.")
    print("Different models have different layer naming conventions!")
