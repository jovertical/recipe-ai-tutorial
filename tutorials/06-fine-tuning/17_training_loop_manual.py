# Problem 17: Manual Training Loop
#
# Implement a training loop from scratch to understand what's happening.
# This demystifies what the HuggingFace Trainer does under the hood.
#
# Training loop steps:
# 1. Forward pass: Get model predictions
# 2. Compute loss: Cross-entropy for language modeling
# 3. Backward pass: Compute gradients
# 4. Update weights: Apply gradients with optimizer
# 5. Repeat for all batches
#
# For LoRA, the key insight is:
# - Only LoRA parameters (A, B) get gradients
# - Base model weights are frozen
#
# ML Relevance: Understanding the training loop helps you debug issues,
# customize training, and implement advanced techniques.

import numpy as np


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        logits: Model outputs, shape (batch, seq_len, vocab_size)
        targets: Target token IDs, shape (batch, seq_len)

    Returns:
        Scalar loss value

    For LM, we predict next token, so:
    - Input: tokens[:-1]
    - Target: tokens[1:]
    """
    # Your solution here
    # 1. Apply softmax to logits
    # 2. Get probability of correct token
    # 3. Return -mean(log(prob))
    pass


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def training_step(
    model,
    batch: dict,
    learning_rate: float = 1e-4
) -> dict:
    """
    Perform one training step.

    Args:
        model: Model with forward() and get_lora_gradients() methods
        batch: Dict with 'input_ids' and 'labels'
        learning_rate: Learning rate for update

    Returns:
        Dict with:
        - 'loss': Loss value
        - 'grad_norm': Gradient norm
        - 'updated_params': Number of params updated

    Steps:
    1. Forward pass
    2. Compute loss
    3. Compute gradients (for LoRA params only)
    4. Update LoRA params
    """
    # Your solution here
    pass


def gradient_accumulation_step(
    model,
    batches: list[dict],
    accumulation_steps: int,
    learning_rate: float = 1e-4
) -> dict:
    """
    Accumulate gradients over multiple batches before updating.

    This simulates larger batch sizes when GPU memory is limited.

    Args:
        model: The model
        batches: List of batches to accumulate over
        accumulation_steps: Number of steps to accumulate
        learning_rate: Learning rate

    Returns:
        - 'total_loss': Sum of losses
        - 'avg_loss': Average loss
        - 'effective_batch_size': Total samples processed
    """
    # Your solution here
    pass


def simple_training_loop(
    model,
    train_data: list[dict],
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    log_every: int = 10
) -> dict:
    """
    Complete training loop.

    Args:
        model: The model to train
        train_data: List of training examples
        num_epochs: Number of passes through data
        batch_size: Samples per batch
        learning_rate: Learning rate
        log_every: Log metrics every N steps

    Returns:
        - 'final_loss': Final training loss
        - 'loss_history': List of losses
        - 'total_steps': Total training steps
    """
    # Your solution here
    pass


def compute_gradient_norm(gradients: dict) -> float:
    """
    Compute total gradient norm across all parameters.

    Args:
        gradients: Dict of parameter_name -> gradient array

    Returns:
        L2 norm of concatenated gradients
    """
    # Your solution here
    pass


def clip_gradients(gradients: dict, max_norm: float) -> dict:
    """
    Clip gradients to prevent exploding gradients.

    If gradient norm > max_norm, scale down proportionally.

    Args:
        gradients: Dict of gradients
        max_norm: Maximum allowed norm

    Returns:
        Clipped gradients (same structure as input)
    """
    # Your solution here
    pass


def learning_rate_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int = 100,
    base_lr: float = 1e-4,
    schedule: str = "linear"
) -> float:
    """
    Compute learning rate for current step.

    Schedules:
    - "constant": Always base_lr
    - "linear": Linear warmup then linear decay
    - "cosine": Linear warmup then cosine decay

    Args:
        step: Current training step
        total_steps: Total steps in training
        warmup_steps: Steps for linear warmup
        base_lr: Base learning rate
        schedule: Schedule type

    Returns:
        Learning rate for this step
    """
    # Your solution here
    pass


class SimpleOptimizer:
    """Simple SGD optimizer for demonstration."""

    def __init__(self, parameters: dict, lr: float = 1e-4):
        """
        Args:
            parameters: Dict of param_name -> param_array
            lr: Learning rate
        """
        self.parameters = parameters
        self.lr = lr

    def step(self, gradients: dict) -> None:
        """Update parameters using gradients."""
        # Your solution here
        pass

    def zero_grad(self) -> None:
        """Reset gradients (not needed for our simple implementation)."""
        pass


class AdamOptimizer:
    """Adam optimizer implementation."""

    def __init__(
        self,
        parameters: dict,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adam optimizer.

        Args:
            parameters: Dict of param_name -> param_array
            lr: Learning rate
            betas: (beta1, beta2) for moment estimates
            eps: Epsilon for numerical stability
            weight_decay: L2 regularization coefficient
        """
        # Your solution here
        pass

    def step(self, gradients: dict) -> None:
        """
        Perform Adam update step.

        Adam update:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
        """
        # Your solution here
        pass


# Mock model for testing
class MockLoRAModel:
    """Mock model for testing training loop."""

    def __init__(self, vocab_size: int = 100, d_model: int = 32):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Simplified: just embedding + linear
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.output = np.random.randn(d_model, vocab_size) * 0.02

        # LoRA parameters
        self.lora_A = np.random.randn(4, vocab_size) * 0.01
        self.lora_B = np.zeros((d_model, 4))

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass returning logits."""
        # Embed
        x = self.embedding[input_ids]  # (batch, seq, d_model)
        # Simple output projection
        logits = x @ self.output  # (batch, seq, vocab)
        return logits

    def get_lora_params(self) -> dict:
        return {'lora_A': self.lora_A, 'lora_B': self.lora_B}


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: Softmax
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    s = softmax(x)
    assert np.allclose(s.sum(axis=-1), [1.0, 1.0]), "Test 1 failed: softmax should sum to 1"

    # Test 2: Cross-entropy loss
    logits = np.array([[[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]])  # (1, 2, 3)
    targets = np.array([[2, 0]])  # (1, 2)
    loss = cross_entropy_loss(logits, targets)
    assert isinstance(loss, float), "Test 2a failed: loss should be scalar"
    assert loss > 0, "Test 2b failed: loss should be positive"

    # Test 3: Gradient norm
    grads = {
        'a': np.array([3.0, 4.0]),  # norm = 5
        'b': np.array([0.0, 0.0])
    }
    norm = compute_gradient_norm(grads)
    assert abs(norm - 5.0) < 1e-6, f"Test 3 failed: expected 5.0, got {norm}"

    # Test 4: Clip gradients
    grads = {'a': np.array([3.0, 4.0])}  # norm = 5
    clipped = clip_gradients(grads, max_norm=2.5)
    new_norm = compute_gradient_norm(clipped)
    assert abs(new_norm - 2.5) < 1e-6, f"Test 4 failed: expected 2.5, got {new_norm}"

    # Test 5: No clipping when under threshold
    grads = {'a': np.array([1.0, 1.0])}  # norm ≈ 1.41
    clipped = clip_gradients(grads, max_norm=5.0)
    assert np.allclose(grads['a'], clipped['a']), "Test 5 failed: should not clip"

    # Test 6: Learning rate schedule - constant
    lr = learning_rate_schedule(50, 100, schedule="constant", base_lr=0.001)
    assert lr == 0.001, f"Test 6 failed: constant should be 0.001, got {lr}"

    # Test 7: Learning rate schedule - warmup
    lr_start = learning_rate_schedule(0, 100, warmup_steps=10, schedule="linear")
    lr_mid = learning_rate_schedule(5, 100, warmup_steps=10, schedule="linear")
    lr_warmup_end = learning_rate_schedule(10, 100, warmup_steps=10, schedule="linear")
    assert lr_start < lr_mid < lr_warmup_end, "Test 7 failed: warmup should increase LR"

    # Test 8: Learning rate schedule - decay
    lr_after_warmup = learning_rate_schedule(20, 100, warmup_steps=10, schedule="linear")
    lr_end = learning_rate_schedule(100, 100, warmup_steps=10, schedule="linear")
    assert lr_after_warmup > lr_end, "Test 8 failed: should decay after warmup"

    # Test 9: Simple optimizer
    params = {'w': np.array([1.0, 2.0])}
    opt = SimpleOptimizer(params, lr=0.1)
    grads = {'w': np.array([1.0, 1.0])}
    opt.step(grads)
    assert np.allclose(params['w'], [0.9, 1.9]), f"Test 9 failed: got {params['w']}"

    # Test 10: Adam optimizer
    params = {'w': np.array([1.0, 2.0])}
    adam = AdamOptimizer(params, lr=0.1)
    grads = {'w': np.array([0.1, 0.1])}
    adam.step(grads)
    # Adam should update parameters
    assert not np.allclose(params['w'], [1.0, 2.0]), "Test 10 failed: Adam should update"

    # Test 11: Mock model forward
    model = MockLoRAModel()
    input_ids = np.array([[1, 2, 3], [4, 5, 6]])
    logits = model.forward(input_ids)
    assert logits.shape == (2, 3, 100), f"Test 11 failed: expected (2, 3, 100), got {logits.shape}"

    # Test 12: Training step (simplified test)
    batch = {
        'input_ids': np.array([[1, 2, 3, 4]]),
        'labels': np.array([[2, 3, 4, 5]])
    }
    result = training_step(model, batch, learning_rate=0.01)
    assert 'loss' in result, "Test 12 failed: should return loss"

    # Test 13: Gradient accumulation
    batches = [batch, batch]
    result = gradient_accumulation_step(model, batches, accumulation_steps=2)
    assert 'avg_loss' in result, "Test 13 failed: should return avg_loss"

    # Test 14: Simple training loop
    train_data = [batch] * 10
    result = simple_training_loop(model, train_data, num_epochs=1, batch_size=1)
    assert 'loss_history' in result, "Test 14a failed: should track loss"
    assert len(result['loss_history']) > 0, "Test 14b failed: should have losses"

    print("All tests passed!")
    print("\nYou now understand the training loop internals!")
    print("The HuggingFace Trainer does all this for you, but knowing how it works helps debug issues.")
