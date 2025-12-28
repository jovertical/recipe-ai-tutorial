# Problem 21: Generation Parameters
#
# Control how your model generates text.
# Different parameters produce dramatically different outputs.
#
# Key parameters:
# - max_new_tokens: How much to generate
# - temperature: Randomness (0=deterministic, 1=normal, >1=creative)
# - top_k: Only consider top k tokens
# - top_p: Only consider tokens with cumulative prob >= p
# - repetition_penalty: Discourage repeating tokens
# - do_sample: Whether to sample or use greedy
#
# ML Relevance: Generation quality depends heavily on these settings.
# Different tasks need different parameters!

import numpy as np


def greedy_decode_step(logits: np.ndarray) -> int:
    """
    Select the most likely token (greedy).

    Args:
        logits: Logits for all vocab tokens, shape (vocab_size,)

    Returns:
        Token ID with highest logit
    """
    # Your solution here
    pass


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to logits.

    Higher temperature = more uniform distribution = more random
    Lower temperature = more peaked distribution = more deterministic

    Formula: scaled_logits = logits / temperature

    Args:
        logits: Raw logits
        temperature: Temperature value (must be > 0)

    Returns:
        Scaled logits
    """
    # Your solution here
    pass


def apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Zero out all but the top k logits.

    Args:
        logits: Raw logits
        k: Number of top tokens to keep

    Returns:
        Filtered logits (non-top-k set to -inf)
    """
    # Your solution here
    pass


def apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus sampling: keep smallest set with cumulative prob >= p.

    Args:
        logits: Raw logits
        p: Cumulative probability threshold (0.9 is common)

    Returns:
        Filtered logits (tokens outside nucleus set to -inf)

    Steps:
    1. Convert to probabilities
    2. Sort by probability
    3. Find cumulative sum
    4. Keep tokens until cumsum >= p
    5. Set others to -inf
    """
    # Your solution here
    pass


def apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: list[int],
    penalty: float = 1.2
) -> np.ndarray:
    """
    Penalize tokens that have already been generated.

    For each token in generated_ids:
    - If logit > 0: divide by penalty
    - If logit < 0: multiply by penalty

    Args:
        logits: Current logits
        generated_ids: Previously generated token IDs
        penalty: Penalty factor (>1 = penalize repetition)

    Returns:
        Adjusted logits
    """
    # Your solution here
    pass


def sample_token(logits: np.ndarray) -> int:
    """
    Sample a token from logits using multinomial sampling.

    Args:
        logits: Logits (will be converted to probabilities)

    Returns:
        Sampled token ID
    """
    # Your solution here
    pass


def generate_with_params(
    model,
    input_ids: np.ndarray,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    eos_token_id: int = None
) -> np.ndarray:
    """
    Generate tokens with all parameters.

    Args:
        model: Model with forward(input_ids) -> logits
        input_ids: Starting tokens, shape (1, seq_len)
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalty for repetition (1.0 = disabled)
        do_sample: If False, use greedy decoding
        eos_token_id: Stop generation at this token

    Returns:
        Generated token IDs including input
    """
    # Your solution here
    pass


def beam_search(
    model,
    input_ids: np.ndarray,
    num_beams: int = 4,
    max_new_tokens: int = 50,
    eos_token_id: int = None
) -> np.ndarray:
    """
    Generate using beam search (explores multiple paths).

    Beam search keeps track of the k best partial sequences.
    More deterministic than sampling, often higher quality for short generations.

    Args:
        model: Model
        input_ids: Starting tokens
        num_beams: Number of beams to track
        max_new_tokens: Maximum generation length
        eos_token_id: End of sequence token

    Returns:
        Best sequence found
    """
    # Your solution here
    pass


def get_recommended_params(task: str) -> dict:
    """
    Get recommended generation parameters for different tasks.

    Args:
        task: One of:
            - "recipe_generation": Creative recipe writing
            - "recipe_completion": Complete a partial recipe
            - "substitution": Suggest ingredient substitutions
            - "chat": Conversational responses
            - "factual": Factual/precise responses

    Returns:
        Dict of recommended parameters
    """
    # Your solution here
    pass


def compare_generation_params(
    model,
    tokenizer,
    prompt: str,
    param_sets: list[dict]
) -> list[dict]:
    """
    Compare outputs with different parameter settings.

    Args:
        model: Model
        tokenizer: Tokenizer
        prompt: Input prompt
        param_sets: List of parameter dicts to compare

    Returns:
        List of dicts with 'params', 'output', 'token_count'
    """
    # Your solution here
    pass


# Mock model for testing
class MockGenerationModel:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        """Return random logits for next token."""
        batch_size = input_ids.shape[0]
        return np.random.randn(batch_size, self.vocab_size)


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: Greedy decode
    logits = np.array([1.0, 3.0, 2.0, 0.5])
    token = greedy_decode_step(logits)
    assert token == 1, f"Test 1 failed: expected 1, got {token}"

    # Test 2: Temperature scaling
    logits = np.array([1.0, 2.0, 3.0])
    scaled_low = apply_temperature(logits, 0.5)
    scaled_high = apply_temperature(logits, 2.0)
    # Lower temp = more peaked = larger difference
    assert (scaled_low[2] - scaled_low[0]) > (logits[2] - logits[0]), \
        "Test 2a failed: low temp should increase difference"
    assert (scaled_high[2] - scaled_high[0]) < (logits[2] - logits[0]), \
        "Test 2b failed: high temp should decrease difference"

    # Test 3: Top-k filtering
    logits = np.array([1.0, 5.0, 3.0, 4.0, 2.0])
    filtered = apply_top_k(logits, k=2)
    assert filtered[1] == 5.0, "Test 3a failed: top token should be kept"
    assert filtered[3] == 4.0, "Test 3b failed: second top should be kept"
    assert filtered[0] == -np.inf, "Test 3c failed: non-top should be -inf"

    # Test 4: Top-p filtering
    logits = np.array([0.0, 2.0, 1.0])  # After softmax: ~[0.09, 0.67, 0.24]
    filtered = apply_top_p(logits, p=0.9)
    # Top 2 tokens (0.67 + 0.24 = 0.91 > 0.9) should be kept
    assert filtered[1] != -np.inf, "Test 4a failed: top token should be kept"
    # The smallest might be filtered out
    assert np.sum(filtered != -np.inf) <= 3, "Test 4b failed: some should be filtered"

    # Test 5: Repetition penalty
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    generated = [1, 3]  # Penalize tokens 1 and 3
    penalized = apply_repetition_penalty(logits, generated, penalty=2.0)
    assert penalized[1] < logits[1], "Test 5a failed: positive logit should decrease"
    assert penalized[3] < logits[3], "Test 5b failed: positive logit should decrease"
    assert penalized[0] == logits[0], "Test 5c failed: non-generated should be unchanged"

    # Test 6: Sample token
    logits = np.array([10.0, -10.0, -10.0])  # Strongly prefer first token
    samples = [sample_token(logits) for _ in range(10)]
    assert all(s == 0 for s in samples), "Test 6 failed: should strongly prefer token 0"

    # Test 7: Recommended params
    for task in ["recipe_generation", "factual", "chat"]:
        params = get_recommended_params(task)
        assert 'temperature' in params, f"Test 7a failed: {task} missing temperature"
        assert 'do_sample' in params, f"Test 7b failed: {task} missing do_sample"

    # Test 8: Different tasks have different params
    creative = get_recommended_params("recipe_generation")
    factual = get_recommended_params("factual")
    assert creative['temperature'] > factual['temperature'], \
        "Test 8 failed: creative should have higher temperature"

    # Test 9: Full generation
    model = MockGenerationModel()
    input_ids = np.array([[1, 2, 3]])
    output = generate_with_params(
        model, input_ids,
        max_new_tokens=10,
        temperature=1.0,
        do_sample=True
    )
    assert output.shape[1] == 13, f"Test 9 failed: expected 13 tokens, got {output.shape[1]}"

    # Test 10: EOS stopping
    model = MockGenerationModel()

    class EosModel:
        def __init__(self):
            self.call_count = 0
            self.vocab_size = 100

        def forward(self, input_ids):
            self.call_count += 1
            logits = np.full((1, 100), -10.0)
            if self.call_count >= 3:
                logits[0, 1] = 10.0  # EOS token
            else:
                logits[0, 50] = 10.0  # Regular token
            return logits

    eos_model = EosModel()
    output = generate_with_params(
        eos_model, np.array([[1]]),
        max_new_tokens=20,
        do_sample=False,
        eos_token_id=1
    )
    # Should stop at EOS (after ~3 tokens)
    assert output.shape[1] <= 5, f"Test 10 failed: should stop at EOS, got {output.shape[1]} tokens"

    print("All tests passed!")
    print("\nGeneration parameters dramatically affect output quality.")
    print("Key settings:")
    print("- Creative tasks: temp=0.7-1.0, top_p=0.9")
    print("- Factual tasks: temp=0.1-0.3, or greedy")
    print("- Avoid repetition: repetition_penalty=1.1-1.3")
