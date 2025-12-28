"""
Exercise 16: Training Configuration

In this exercise, you'll learn to set up training configurations for
fine-tuning with LoRA. Proper configuration is crucial for:
1. Memory efficiency (fitting in GPU/MPS memory)
2. Training stability (appropriate learning rates, warmup)
3. Convergence speed (batch size, gradient accumulation)

Example:
    # LoRA configuration
    lora_config = {
        "r": 8,              # Low rank
        "lora_alpha": 16,    # Scaling factor
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05
    }
    
    # Training arguments
    training_args = {
        "learning_rate": 2e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,  # Effective batch = 16
        "num_epochs": 3,
        "warmup_ratio": 0.03
    }

ML Relevance:
    Training configuration can make or break your fine-tuning:
    - Too high learning rate → training diverges
    - Too low → training takes forever or gets stuck
    - Wrong batch size → OOM errors or poor gradient estimates
    - No warmup → unstable early training
    
    These exercises help you understand the trade-offs.

Your Task:
    1. Implement create_lora_config() - set up LoRA parameters
    2. Implement create_training_args() - set up training hyperparameters
    3. Implement calculate_effective_batch_size() - understand gradient accumulation
    4. Implement estimate_training_time() - plan your training runs
    5. Implement create_optimizer_config() - optimizer and scheduler setup

Run:
    python tutorials/06-fine-tuning/16_training_config.py
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import math


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    r: int = 8                          # Rank
    lora_alpha: int = 16                # Scaling = alpha / r
    target_modules: List[str] = None    # Which modules to adapt
    lora_dropout: float = 0.05          # Dropout on LoRA layers
    bias: str = "none"                  # "none", "all", or "lora_only"
    task_type: str = "CAUSAL_LM"        # Task type
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


@dataclass  
class TrainingArgs:
    """Configuration for training."""
    learning_rate: float = 2e-4
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_steps: int = -1                 # -1 means use epochs
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False


def create_lora_config(
    rank: int = 8,
    alpha: int = None,
    target_modules: List[str] = None,
    dropout: float = 0.05,
    task_type: str = "CAUSAL_LM"
) -> Dict[str, Any]:
    """
    Create a LoRA configuration dictionary.
    
    Args:
        rank: LoRA rank (lower = fewer parameters, higher = more capacity)
        alpha: Scaling factor (defaults to 2 * rank if None)
        target_modules: Which modules to apply LoRA to
        dropout: Dropout probability for LoRA layers
        task_type: "CAUSAL_LM", "SEQ_CLS", etc.
        
    Returns:
        Dictionary with LoRA configuration
        
    Example:
        >>> config = create_lora_config(rank=8)
        >>> config["r"]
        8
        >>> config["lora_alpha"]
        16
        >>> config["scaling"]  # alpha / r
        2.0
    """
    # Your solution here
    # Hints:
    # 1. Default alpha to 2 * rank if not provided
    # 2. Default target_modules to ["q_proj", "v_proj"]
    # 3. Calculate scaling factor as alpha / r
    # 4. Return dict with all parameters
    pass


def create_training_args(
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    use_fp16: bool = False,
    use_bf16: bool = False
) -> Dict[str, Any]:
    """
    Create training arguments dictionary.
    
    Args:
        learning_rate: Learning rate for optimizer
        batch_size: Per-device batch size
        gradient_accumulation_steps: Steps to accumulate before update
        num_epochs: Number of training epochs
        warmup_ratio: Fraction of steps for warmup
        weight_decay: Weight decay for regularization
        use_fp16: Use 16-bit floating point (NVIDIA GPUs)
        use_bf16: Use bfloat16 (newer NVIDIA GPUs, Apple M-series)
        
    Returns:
        Dictionary with training arguments
        
    Example:
        >>> args = create_training_args(batch_size=4, gradient_accumulation_steps=8)
        >>> args["effective_batch_size"]
        32
    """
    # Your solution here
    # Hints:
    # 1. Calculate effective batch size
    # 2. Include all provided parameters
    # 3. Add derived values like effective_batch_size
    pass


def calculate_effective_batch_size(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_devices: int = 1
) -> int:
    """
    Calculate the effective batch size with gradient accumulation.
    
    Gradient accumulation allows simulating larger batches when memory is limited.
    The effective batch size is what the optimizer "sees".
    
    Args:
        per_device_batch_size: Batch size per device per step
        gradient_accumulation_steps: Number of steps before optimizer update
        num_devices: Number of GPUs/devices
        
    Returns:
        Effective batch size
        
    Example:
        >>> calculate_effective_batch_size(4, 8, 1)
        32
        >>> calculate_effective_batch_size(4, 4, 2)  # 2 GPUs
        32
    """
    # Your solution here
    pass


def calculate_training_steps(
    num_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int
) -> Dict[str, int]:
    """
    Calculate the number of training steps.
    
    Args:
        num_examples: Total number of training examples
        batch_size: Per-device batch size
        gradient_accumulation_steps: Accumulation steps
        num_epochs: Number of epochs
        
    Returns:
        Dictionary with:
        - steps_per_epoch: Optimizer updates per epoch
        - total_steps: Total optimizer updates
        - warmup_steps: Steps for warmup (3% of total)
        
    Example:
        >>> result = calculate_training_steps(1000, 4, 4, 3)
        >>> result["steps_per_epoch"]
        62  # ceil(1000 / (4 * 4)) = ceil(62.5)
        >>> result["total_steps"]
        186  # 62 * 3
    """
    # Your solution here
    # Hints:
    # 1. Effective batch = batch_size * gradient_accumulation_steps
    # 2. steps_per_epoch = ceil(num_examples / effective_batch)
    # 3. total_steps = steps_per_epoch * num_epochs
    # 4. warmup_steps = int(total_steps * 0.03)
    pass


def estimate_training_time(
    total_steps: int,
    seconds_per_step: float = 1.0,
    eval_steps: int = 100,
    seconds_per_eval: float = 30.0
) -> Dict[str, float]:
    """
    Estimate total training time.
    
    Args:
        total_steps: Total optimizer steps
        seconds_per_step: Estimated time per training step
        eval_steps: How often to evaluate
        seconds_per_eval: Time for one evaluation
        
    Returns:
        Dictionary with time estimates in different units
        
    Example:
        >>> result = estimate_training_time(1000, seconds_per_step=0.5)
        >>> result["total_seconds"]
        650.0  # 500 training + 10 evals * 30 = 800? Let me recalculate...
    """
    # Your solution here
    # Hints:
    # 1. Training time = total_steps * seconds_per_step
    # 2. Num evals = total_steps // eval_steps
    # 3. Total = training time + num evals * seconds_per_eval
    pass


def create_optimizer_config(
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    optimizer_type: str = "adamw"
) -> Dict[str, Any]:
    """
    Create optimizer configuration.
    
    Args:
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        beta1: Adam beta1 (momentum)
        beta2: Adam beta2 (RMSprop-like)
        epsilon: Numerical stability
        optimizer_type: "adamw", "adam", "sgd"
        
    Returns:
        Dictionary with optimizer configuration
        
    Example:
        >>> config = create_optimizer_config(learning_rate=1e-4)
        >>> config["optimizer_type"]
        'adamw'
    """
    # Your solution here
    pass


def create_scheduler_config(
    scheduler_type: str = "cosine",
    num_warmup_steps: int = 100,
    num_training_steps: int = 1000,
    min_lr_ratio: float = 0.1
) -> Dict[str, Any]:
    """
    Create learning rate scheduler configuration.
    
    Common scheduler types:
    - "linear": Linear decay from max to 0
    - "cosine": Cosine annealing
    - "constant": No decay
    - "constant_with_warmup": Warmup then constant
    
    Args:
        scheduler_type: Type of LR scheduler
        num_warmup_steps: Steps for warmup
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of initial (for cosine)
        
    Returns:
        Dictionary with scheduler configuration
        
    Example:
        >>> config = create_scheduler_config("cosine", 100, 1000)
        >>> config["scheduler_type"]
        'cosine'
    """
    # Your solution here
    pass


def recommend_config_for_model(
    model_size_b: float,
    available_memory_gb: float,
    dataset_size: int
) -> Dict[str, Any]:
    """
    Recommend training configuration based on resources.
    
    This is a heuristic-based recommendation.
    
    Args:
        model_size_b: Model size in billions of parameters
        available_memory_gb: Available GPU/MPS memory in GB
        dataset_size: Number of training examples
        
    Returns:
        Recommended configuration dictionary
        
    Example:
        >>> config = recommend_config_for_model(1.1, 16, 10000)
        >>> config["lora_rank"]  # Suggested LoRA rank
        8
        >>> config["batch_size"]  # Suggested batch size
        4
    """
    # Your solution here
    # Hints (rough rules of thumb):
    # 1. Smaller models can use higher ranks
    # 2. Less memory = smaller batch, more accumulation
    # 3. Larger datasets = can use lower learning rate
    # 4. Use bf16 if available (M-series Macs)
    pass


def validate_config(
    lora_config: Dict[str, Any],
    training_args: Dict[str, Any]
) -> List[str]:
    """
    Validate configuration and return warnings.
    
    Args:
        lora_config: LoRA configuration
        training_args: Training arguments
        
    Returns:
        List of warning messages (empty if all good)
        
    Example:
        >>> lora = {"r": 64, "lora_alpha": 8}
        >>> train = {"learning_rate": 0.1}
        >>> warnings = validate_config(lora, train)
        >>> len(warnings) > 0
        True  # High LR and unusual alpha/r ratio
    """
    # Your solution here
    # Check for common issues:
    # 1. Learning rate too high (> 1e-3)
    # 2. Alpha < rank (unusual)
    # 3. Very high rank (> 64)
    # 4. Very low warmup ratio with high LR
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    print("Testing create_lora_config...")
    
    # Test 1: Default config
    config = create_lora_config(rank=8)
    assert config["r"] == 8, "Test 1a failed"
    assert config["lora_alpha"] == 16, f"Test 1b failed: expected 16, got {config['lora_alpha']}"
    assert config["scaling"] == 2.0, f"Test 1c failed: expected 2.0, got {config['scaling']}"
    assert "q_proj" in config["target_modules"], "Test 1d failed"
    print("  ✓ Default config correct")
    
    # Test 2: Custom config
    config = create_lora_config(rank=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj"])
    assert config["r"] == 16, "Test 2a failed"
    assert config["lora_alpha"] == 32, "Test 2b failed"
    assert len(config["target_modules"]) == 3, "Test 2c failed"
    print("  ✓ Custom config correct")
    
    print("\nTesting create_training_args...")
    
    # Test 3: Default args
    args = create_training_args()
    assert args["learning_rate"] == 2e-4, "Test 3a failed"
    assert "effective_batch_size" in args, "Test 3b failed"
    print("  ✓ Default args correct")
    
    # Test 4: Effective batch size in args
    args = create_training_args(batch_size=4, gradient_accumulation_steps=8)
    assert args["effective_batch_size"] == 32, f"Test 4 failed: got {args['effective_batch_size']}"
    print("  ✓ Effective batch size calculated")
    
    print("\nTesting calculate_effective_batch_size...")
    
    # Test 5: Single device
    result = calculate_effective_batch_size(4, 8, 1)
    assert result == 32, f"Test 5a failed: got {result}"
    
    # Test 6: Multiple devices
    result = calculate_effective_batch_size(4, 4, 2)
    assert result == 32, f"Test 6 failed: got {result}"
    print("  ✓ Effective batch size calculation correct")
    
    print("\nTesting calculate_training_steps...")
    
    # Test 7: Training steps
    result = calculate_training_steps(1000, 4, 4, 3)
    assert result["steps_per_epoch"] == 63, f"Test 7a failed: got {result['steps_per_epoch']}"
    assert result["total_steps"] == 189, f"Test 7b failed: got {result['total_steps']}"
    assert "warmup_steps" in result, "Test 7c failed"
    print("  ✓ Training steps calculated correctly")
    
    # Test 8: Exact division
    result = calculate_training_steps(160, 8, 2, 2)  # 160 / 16 = 10 exact
    assert result["steps_per_epoch"] == 10, f"Test 8a failed: got {result['steps_per_epoch']}"
    assert result["total_steps"] == 20, f"Test 8b failed: got {result['total_steps']}"
    print("  ✓ Exact division handled")
    
    print("\nTesting estimate_training_time...")
    
    # Test 9: Time estimation
    result = estimate_training_time(1000, seconds_per_step=0.5, eval_steps=100, seconds_per_eval=30.0)
    assert "total_seconds" in result, "Test 9a failed"
    assert "total_minutes" in result, "Test 9b failed"
    assert "total_hours" in result, "Test 9c failed"
    training_time = 1000 * 0.5
    num_evals = 1000 // 100
    expected_total = training_time + num_evals * 30.0
    assert abs(result["total_seconds"] - expected_total) < 1, f"Test 9d failed: {result['total_seconds']} vs {expected_total}"
    print("  ✓ Time estimation correct")
    
    print("\nTesting create_optimizer_config...")
    
    # Test 10: Optimizer config
    config = create_optimizer_config(learning_rate=1e-4)
    assert config["learning_rate"] == 1e-4, "Test 10a failed"
    assert config["optimizer_type"] == "adamw", "Test 10b failed"
    assert "beta1" in config, "Test 10c failed"
    assert "beta2" in config, "Test 10d failed"
    print("  ✓ Optimizer config correct")
    
    print("\nTesting create_scheduler_config...")
    
    # Test 11: Scheduler config
    config = create_scheduler_config("cosine", 100, 1000)
    assert config["scheduler_type"] == "cosine", "Test 11a failed"
    assert config["num_warmup_steps"] == 100, "Test 11b failed"
    assert config["num_training_steps"] == 1000, "Test 11c failed"
    print("  ✓ Scheduler config correct")
    
    print("\nTesting recommend_config_for_model...")
    
    # Test 12: Small model, good memory
    config = recommend_config_for_model(1.1, 16, 10000)
    assert "lora_rank" in config, "Test 12a failed"
    assert "batch_size" in config, "Test 12b failed"
    assert "learning_rate" in config, "Test 12c failed"
    print("  ✓ Recommendation includes required fields")
    
    # Test 13: Limited memory should suggest smaller batch
    config_limited = recommend_config_for_model(1.1, 8, 10000)
    config_plenty = recommend_config_for_model(1.1, 32, 10000)
    # Limited memory should have smaller batch or more accumulation
    effective_limited = config_limited.get("batch_size", 4) * config_limited.get("gradient_accumulation_steps", 1)
    effective_plenty = config_plenty.get("batch_size", 4) * config_plenty.get("gradient_accumulation_steps", 1)
    # Just check that both return valid configs
    assert config_limited["batch_size"] > 0, "Test 13a failed"
    assert config_plenty["batch_size"] > 0, "Test 13b failed"
    print("  ✓ Memory-aware recommendations")
    
    print("\nTesting validate_config...")
    
    # Test 14: Valid config
    lora = {"r": 8, "lora_alpha": 16}
    train = {"learning_rate": 2e-4, "warmup_ratio": 0.03}
    warnings = validate_config(lora, train)
    assert isinstance(warnings, list), "Test 14a failed"
    print("  ✓ Validation returns list")
    
    # Test 15: Invalid config (high LR)
    train_bad = {"learning_rate": 0.5, "warmup_ratio": 0.0}
    warnings = validate_config(lora, train_bad)
    assert len(warnings) > 0, "Test 15 failed: should warn about high LR"
    print("  ✓ Warns about bad config")
    
    # Test 16: Unusual alpha/r ratio
    lora_unusual = {"r": 64, "lora_alpha": 8}
    warnings = validate_config(lora_unusual, train)
    has_warning = len(warnings) > 0
    print(f"  ✓ Unusual config detected: {has_warning}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    print("\nYou've learned to configure training for LoRA fine-tuning.")
    print("Key takeaways:")
    print("- LoRA rank controls capacity vs. efficiency trade-off")
    print("- Gradient accumulation simulates larger batches")
    print("- Warmup prevents early training instability")
    print("- Configuration validation catches common mistakes")
