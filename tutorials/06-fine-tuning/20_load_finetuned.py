# Problem 20: Loading Fine-tuned Models
#
# Learn how to properly load fine-tuned models with LoRA adapters.
# This includes:
# 1. Loading the base model
# 2. Loading and applying LoRA adapters
# 3. Optionally merging adapters into base weights
# 4. Preparing for inference
#
# Example:
#   # Load base model
#   base_model = load_base_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
#
#   # Load LoRA adapter
#   model = load_adapter(base_model, "path/to/adapter")
#
#   # Optionally merge for faster inference
#   model = merge_and_unload(model)
#
#   # Generate
#   output = generate(model, "How do I make pasta?")
#
# ML Relevance: Understanding model loading is crucial for deployment:
# - LoRA keeps base model separate -> easy to swap adapters
# - Merging speeds up inference but loses flexibility
# - Memory management matters for deployment
# - Different loading strategies for different use cases

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import json


@dataclass
class MockLinearLayer:
    """Mock linear layer for testing."""
    weight: np.ndarray
    bias: Optional[np.ndarray] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


@dataclass
class MockLoRALayer:
    """
    Mock LoRA layer that wraps a linear layer.

    The output is: base_output + (x @ A.T @ B.T) * scaling
    where scaling = alpha / rank
    """
    base_layer: MockLinearLayer
    lora_A: np.ndarray  # Shape: (rank, in_features)
    lora_B: np.ndarray  # Shape: (out_features, rank)
    alpha: float = 16.0
    rank: int = 8

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with LoRA."""
        # Your solution here
        pass

    def merge_weights(self) -> MockLinearLayer:
        """
        Merge LoRA weights into base layer.

        Returns:
            New MockLinearLayer with merged weights
        """
        # Your solution here
        pass


class MockBaseModel:
    """
    Mock base model for testing loading logic.

    This simulates a transformer model with attention layers.
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2):
        """
        Initialize mock model.

        Args:
            hidden_size: Hidden dimension
            num_layers: Number of transformer layers
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.config = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "model_type": "mock_llama"
        }

        # Create mock attention layers
        self.layers = {}
        for i in range(num_layers):
            self.layers[f"layer_{i}"] = {
                "q_proj": MockLinearLayer(np.random.randn(hidden_size, hidden_size)),
                "k_proj": MockLinearLayer(np.random.randn(hidden_size, hidden_size)),
                "v_proj": MockLinearLayer(np.random.randn(hidden_size, hidden_size)),
                "o_proj": MockLinearLayer(np.random.randn(hidden_size, hidden_size)),
            }

        self.is_quantized = False
        self.device = "cpu"

    def get_layer(self, layer_path: str) -> MockLinearLayer:
        """Get a layer by path like 'layer_0.q_proj'."""
        parts = layer_path.split(".")
        layer_idx = parts[0]
        proj_name = parts[1]
        return self.layers[layer_idx][proj_name]

    def set_layer(self, layer_path: str, new_layer):
        """Set a layer by path."""
        parts = layer_path.split(".")
        layer_idx = parts[0]
        proj_name = parts[1]
        self.layers[layer_idx][proj_name] = new_layer

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Simple forward pass."""
        for layer_name in sorted(self.layers.keys()):
            layer = self.layers[layer_name]
            # Simplified attention: just sum all projections
            q = layer["q_proj"](x)
            k = layer["k_proj"](x)
            v = layer["v_proj"](x)
            x = layer["o_proj"](q + k + v)
        return x


class MockLoRAModel:
    """
    Mock model with LoRA adapters applied.
    """

    def __init__(
        self,
        base_model: MockBaseModel,
        lora_config: Dict[str, Any]
    ):
        """
        Initialize LoRA model.

        Args:
            base_model: The base model to adapt
            lora_config: LoRA configuration
        """
        # Your solution here
        pass

    def add_adapter(
        self,
        layer_path: str,
        lora_A: np.ndarray,
        lora_B: np.ndarray
    ) -> None:
        """
        Add LoRA adapter to a layer.

        Args:
            layer_path: Path to layer (e.g., "layer_0.q_proj")
            lora_A: A matrix (rank, in_features)
            lora_B: B matrix (out_features, rank)
        """
        # Your solution here
        pass

    def get_trainable_params(self) -> int:
        """Count trainable (LoRA) parameters."""
        # Your solution here
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using LoRA layers where available."""
        # Your solution here
        pass


def load_base_model(
    model_name: str,
    device: str = "cpu",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> MockBaseModel:
    """
    Load a base model (mock implementation).

    In real code, this would use transformers.AutoModelForCausalLM.from_pretrained()

    Args:
        model_name: Model name or path
        device: Device to load on ("cpu", "cuda", "mps")
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization (QLoRA)

    Returns:
        Loaded base model

    Example:
        >>> model = load_base_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> model.config["model_type"]
        'mock_llama'
    """
    # Your solution here
    pass


def load_adapter(
    base_model: MockBaseModel,
    adapter_path: str,
    adapter_name: str = "default"
) -> MockLoRAModel:
    """
    Load LoRA adapter and apply to base model.

    In real code, this would use peft.PeftModel.from_pretrained()

    Args:
        base_model: The base model
        adapter_path: Path to adapter weights
        adapter_name: Name for the adapter

    Returns:
        Model with LoRA adapter applied

    Example:
        >>> base = load_base_model("TinyLlama")
        >>> model = load_adapter(base, "path/to/adapter")
        >>> model.get_trainable_params() > 0
        True
    """
    # Your solution here
    pass


def merge_and_unload(lora_model: MockLoRAModel) -> MockBaseModel:
    """
    Merge LoRA weights into base model and return base model only.

    This is useful for faster inference when you don't need to swap adapters.

    Args:
        lora_model: Model with LoRA adapters

    Returns:
        Base model with LoRA weights merged in

    Example:
        >>> lora_model = load_adapter(base, "adapter")
        >>> merged = merge_and_unload(lora_model)
        >>> type(merged).__name__
        'MockBaseModel'
    """
    # Your solution here
    pass


def prepare_for_inference(
    model: MockBaseModel,
    use_cache: bool = True,
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """
    Prepare model for inference.

    Args:
        model: The model to prepare
        use_cache: Enable KV cache for faster generation
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with inference configuration

    Example:
        >>> config = prepare_for_inference(model, max_new_tokens=100)
        >>> config["max_new_tokens"]
        100
    """
    # Your solution here
    pass


def save_adapter(
    lora_model: MockLoRAModel,
    save_path: str
) -> Dict[str, Any]:
    """
    Save LoRA adapter weights.

    Args:
        lora_model: Model with LoRA adapters
        save_path: Path to save adapter

    Returns:
        Dictionary describing what was saved

    Example:
        >>> result = save_adapter(model, "/path/to/save")
        >>> result["num_adapters"]
        4
    """
    # Your solution here
    pass


def list_adapters(adapter_path: str) -> List[str]:
    """
    List available adapters in a path.

    Args:
        adapter_path: Path containing adapters

    Returns:
        List of adapter names

    Example:
        >>> adapters = list_adapters("/path/to/adapters")
        >>> "default" in adapters
        True
    """
    # Your solution here (mock implementation)
    pass


def compare_outputs(
    base_model: MockBaseModel,
    lora_model: MockLoRAModel,
    input_data: np.ndarray
) -> Dict[str, Any]:
    """
    Compare outputs between base and LoRA model.

    Args:
        base_model: Original base model
        lora_model: Model with LoRA adapters
        input_data: Input to compare on

    Returns:
        Dictionary with comparison metrics

    Example:
        >>> result = compare_outputs(base, lora, np.random.randn(1, 64))
        >>> "output_difference" in result
        True
    """
    # Your solution here
    pass


def memory_footprint(
    model: MockBaseModel,
    adapter: MockLoRAModel = None
) -> Dict[str, float]:
    """
    Estimate memory footprint of model components.

    Args:
        model: Base model
        adapter: Optional LoRA adapter

    Returns:
        Dictionary with memory estimates in MB

    Example:
        >>> mem = memory_footprint(base_model, lora_model)
        >>> mem["base_model_mb"] > 0
        True
    """
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    print("Testing MockLoRALayer...")

    # Test 1: LoRA layer forward
    base = MockLinearLayer(np.random.randn(32, 64))
    lora_A = np.random.randn(8, 64) * 0.01
    lora_B = np.random.randn(32, 8) * 0.01
    lora_layer = MockLoRALayer(base, lora_A, lora_B, alpha=16.0, rank=8)

    x = np.random.randn(2, 64)
    output = lora_layer(x)
    assert output.shape == (2, 32), f"Test 1a failed: {output.shape}"

    # Output should be different from base
    base_output = base(x)
    assert not np.allclose(output, base_output), "Test 1b failed: LoRA should change output"
    print("  ✓ LoRA layer forward works")

    # Test 2: Merge weights
    merged = lora_layer.merge_weights()
    merged_output = merged(x)
    assert np.allclose(output, merged_output, atol=1e-5), "Test 2 failed: merged should match"
    print("  ✓ Weight merging works")

    print("\nTesting MockBaseModel...")

    # Test 3: Base model creation
    model = MockBaseModel(hidden_size=64, num_layers=2)
    assert len(model.layers) == 2, "Test 3a failed"
    assert "q_proj" in model.layers["layer_0"], "Test 3b failed"
    print("  ✓ Base model created")

    # Test 4: Forward pass
    x = np.random.randn(2, 64)
    output = model(x)
    assert output.shape == (2, 64), f"Test 4 failed: {output.shape}"
    print("  ✓ Base model forward works")

    print("\nTesting MockLoRAModel...")

    # Test 5: LoRA model creation
    lora_config = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}
    lora_model = MockLoRAModel(model, lora_config)
    assert lora_model is not None, "Test 5 failed"
    print("  ✓ LoRA model created")

    # Test 6: Add adapter
    for i in range(2):
        for proj in ["q_proj", "v_proj"]:
            lora_A = np.random.randn(8, 64) * 0.01
            lora_B = np.random.randn(64, 8) * 0.01
            lora_model.add_adapter(f"layer_{i}.{proj}", lora_A, lora_B)

    trainable = lora_model.get_trainable_params()
    expected = 4 * (8 * 64 + 64 * 8)  # 4 adapters, each has A and B
    assert trainable == expected, f"Test 6 failed: {trainable} vs {expected}"
    print("  ✓ Adapters added correctly")

    # Test 7: LoRA forward pass
    x = np.random.randn(2, 64)
    lora_output = lora_model(x)
    assert lora_output.shape == (2, 64), f"Test 7 failed: {lora_output.shape}"
    print("  ✓ LoRA model forward works")

    print("\nTesting load_base_model...")

    # Test 8: Load base model
    model = load_base_model("mock-model", device="cpu")
    assert model is not None, "Test 8a failed"
    assert model.device == "cpu", f"Test 8b failed: {model.device}"
    print("  ✓ Base model loaded")

    # Test 9: Quantization flags
    model = load_base_model("mock-model", load_in_8bit=True)
    assert model.is_quantized, "Test 9 failed"
    print("  ✓ Quantization flag set")

    print("\nTesting load_adapter...")

    # Test 10: Load adapter
    base = load_base_model("mock-model")
    lora = load_adapter(base, "mock/adapter/path")
    assert lora is not None, "Test 10a failed"
    assert lora.get_trainable_params() > 0, "Test 10b failed"
    print("  ✓ Adapter loaded")

    print("\nTesting merge_and_unload...")

    # Test 11: Merge and unload
    base = load_base_model("mock-model")
    x = np.random.randn(2, 64)
    base_before = base(x).copy()

    lora = load_adapter(base, "mock/adapter")
    merged = merge_and_unload(lora)

    assert type(merged).__name__ == "MockBaseModel", f"Test 11a failed: {type(merged)}"
    merged_output = merged(x)
    print("  ✓ Merge and unload works")

    print("\nTesting prepare_for_inference...")

    # Test 12: Prepare for inference
    model = load_base_model("mock-model")
    config = prepare_for_inference(model, max_new_tokens=100)
    assert "max_new_tokens" in config, "Test 12a failed"
    assert config["max_new_tokens"] == 100, "Test 12b failed"
    assert "use_cache" in config, "Test 12c failed"
    print("  ✓ Inference config prepared")

    print("\nTesting save_adapter...")

    # Test 13: Save adapter
    base = load_base_model("mock-model")
    lora = load_adapter(base, "mock/adapter")
    result = save_adapter(lora, "/mock/save/path")
    assert "num_adapters" in result, "Test 13a failed"
    assert result["num_adapters"] > 0, "Test 13b failed"
    print("  ✓ Adapter save info returned")

    print("\nTesting list_adapters...")

    # Test 14: List adapters
    adapters = list_adapters("/mock/path")
    assert isinstance(adapters, list), "Test 14 failed"
    print("  ✓ Adapter listing works")

    print("\nTesting compare_outputs...")

    # Test 15: Compare outputs
    base = load_base_model("mock-model")
    base_copy = load_base_model("mock-model")
    lora = load_adapter(base, "mock/adapter")
    x = np.random.randn(2, 64)

    result = compare_outputs(base_copy, lora, x)
    assert "output_difference" in result, "Test 15 failed"
    print("  ✓ Output comparison works")

    print("\nTesting memory_footprint...")

    # Test 16: Memory footprint
    base = load_base_model("mock-model")
    lora = load_adapter(base, "mock/adapter")
    mem = memory_footprint(base, lora)

    assert "base_model_mb" in mem, "Test 16a failed"
    assert "adapter_mb" in mem, "Test 16b failed"
    assert mem["base_model_mb"] > 0, "Test 16c failed"
    print("  ✓ Memory footprint calculated")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    print("\nKey takeaways:")
    print("- LoRA adapters are loaded on top of frozen base models")
    print("- Merging combines LoRA weights into base for faster inference")
    print("- Memory management is crucial for deployment")
    print("- You can easily swap adapters for different tasks")
