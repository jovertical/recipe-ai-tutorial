# Problem 14: Tokenize Dataset for Training
#
# Prepare your recipe data for language model training.
# This is where data processing meets tokenization.
#
# Key concepts:
# 1. Tokenize with proper padding/truncation
# 2. Create labels for causal LM (labels = input shifted by 1)
# 3. Handle variable-length sequences efficiently
# 4. Ignore padding tokens in loss calculation
#
# For causal LM:
#   Input:  [BOS, t1, t2, t3, t4]
#   Labels: [t1,  t2, t3, t4, EOS]
#   Loss computed only on non-padding tokens
#
# ML Relevance: Bad tokenization = garbage model. This step is critical.

import numpy as np


def tokenize_text(text: str, tokenizer, max_length: int = 512) -> dict:
    """
    Tokenize a single text for training.

    Args:
        text: Input text
        tokenizer: Tokenizer with encode() method
        max_length: Maximum sequence length

    Returns:
        Dict with:
        - 'input_ids': Token IDs
        - 'attention_mask': 1 for real tokens, 0 for padding
        - 'length': Actual length before padding
    """
    # Your solution here
    pass


def create_labels(input_ids: np.ndarray, pad_token_id: int = 0) -> np.ndarray:
    """
    Create labels for causal language modeling.

    Labels are input_ids shifted left by 1.
    Padding positions get label -100 (ignored in loss).

    Args:
        input_ids: Token IDs, shape (seq_len,) or (batch, seq_len)
        pad_token_id: ID of padding token

    Returns:
        Labels array, same shape as input_ids

    Example:
        input_ids = [1, 5, 6, 7, 0, 0]  # 0 is padding
        labels    = [5, 6, 7, -100, -100, -100]
    """
    # Your solution here
    pass


def tokenize_for_causal_lm(
    text: str,
    tokenizer,
    max_length: int = 512,
    add_eos: bool = True
) -> dict:
    """
    Complete tokenization for causal LM training.

    Args:
        text: Input text
        tokenizer: Tokenizer
        max_length: Maximum length (truncate if longer)
        add_eos: Whether to add EOS token at end

    Returns:
        Dict with 'input_ids', 'attention_mask', 'labels'
    """
    # Your solution here
    pass


def tokenize_instruction_pair(
    instruction: str,
    response: str,
    tokenizer,
    max_length: int = 512
) -> dict:
    """
    Tokenize instruction-response pair.

    Only compute loss on the response part, not the instruction.
    This teaches the model to respond, not to generate instructions.

    Args:
        instruction: The prompt/instruction
        response: The target response
        tokenizer: Tokenizer
        max_length: Maximum total length

    Returns:
        Dict with 'input_ids', 'attention_mask', 'labels'
        Labels are -100 for instruction part, actual IDs for response
    """
    # Your solution here
    pass


def batch_tokenize(
    texts: list[str],
    tokenizer,
    max_length: int = 512,
    padding: str = "max_length"
) -> dict:
    """
    Tokenize a batch of texts with padding.

    Args:
        texts: List of texts
        tokenizer: Tokenizer
        max_length: Maximum length
        padding: "max_length" or "longest"

    Returns:
        Dict with batched 'input_ids', 'attention_mask', 'labels'
    """
    # Your solution here
    pass


def chunk_long_text(
    text: str,
    tokenizer,
    chunk_size: int = 512,
    overlap: int = 50
) -> list[str]:
    """
    Split long text into overlapping chunks.

    Useful when text exceeds model's context length.

    Args:
        text: Long input text
        tokenizer: Tokenizer (for accurate token counting)
        chunk_size: Target chunk size in tokens
        overlap: Token overlap between chunks

    Returns:
        List of text chunks
    """
    # Your solution here
    pass


def prepare_recipe_dataset(
    recipes: list[dict],
    tokenizer,
    max_length: int = 512,
    template_fn=None
) -> list[dict]:
    """
    Prepare a full recipe dataset for training.

    Args:
        recipes: List of recipe dicts with 'title', 'ingredients', 'instructions'
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        template_fn: Optional function to format recipe as text

    Returns:
        List of tokenized examples ready for training
    """
    # Your solution here
    pass


def filter_by_length(
    examples: list[dict],
    min_length: int = 10,
    max_length: int = 512
) -> list[dict]:
    """
    Filter examples by token length.

    Removes too-short and too-long examples.

    Returns filtered list and statistics.
    """
    # Your solution here
    pass


def dataset_statistics(examples: list[dict]) -> dict:
    """
    Compute statistics about tokenized dataset.

    Returns:
    - 'num_examples': Count
    - 'avg_length': Average token length
    - 'max_length': Maximum length
    - 'min_length': Minimum length
    - 'length_histogram': Distribution of lengths
    - 'total_tokens': Sum of all tokens
    """
    # Your solution here
    pass


# Mock tokenizer for testing
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        # Simple mock: split on spaces, hash to IDs
        tokens = [hash(w) % 1000 + 3 for w in text.split()]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, ids):
        return f"<decoded {len(ids)} tokens>"


# Sample recipes
SAMPLE_RECIPES = [
    {
        "title": "Simple Pasta",
        "ingredients": ["pasta", "olive oil", "garlic", "parmesan"],
        "instructions": ["Cook pasta", "Sauté garlic in oil", "Toss together", "Top with cheese"]
    },
    {
        "title": "Quick Cookies",
        "ingredients": ["flour", "butter", "sugar", "eggs"],
        "instructions": ["Mix ingredients", "Form balls", "Bake at 350F for 12 minutes"]
    }
]


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    tokenizer = MockTokenizer()

    # Test 1: Basic tokenization
    result = tokenize_text("Hello world", tokenizer, max_length=10)
    assert 'input_ids' in result, "Test 1a failed: missing input_ids"
    assert 'attention_mask' in result, "Test 1b failed: missing attention_mask"
    assert len(result['input_ids']) <= 10, "Test 1c failed: exceeds max_length"

    # Test 2: Create labels
    input_ids = np.array([2, 5, 6, 7, 0, 0])  # BOS, tokens, padding
    labels = create_labels(input_ids, pad_token_id=0)
    assert labels[-1] == -100, "Test 2a failed: padding should be -100"
    assert labels[-2] == -100, "Test 2b failed: padding should be -100"
    # Non-padding labels should be shifted
    assert labels[0] == 5, f"Test 2c failed: expected 5, got {labels[0]}"

    # Test 3: Causal LM tokenization
    result = tokenize_for_causal_lm("Test sentence here", tokenizer)
    assert 'labels' in result, "Test 3a failed: missing labels"
    assert len(result['labels']) == len(result['input_ids']), "Test 3b failed: length mismatch"

    # Test 4: Instruction pair tokenization
    result = tokenize_instruction_pair(
        "Make a cake",
        "Here is how to make a cake",
        tokenizer
    )
    assert 'labels' in result, "Test 4a failed: missing labels"
    # Instruction part should have -100 labels
    assert -100 in result['labels'], "Test 4b failed: instruction should be masked"

    # Test 5: Batch tokenization
    texts = ["Hello", "World here", "Test sentence"]
    batch = batch_tokenize(texts, tokenizer, max_length=20)
    assert batch['input_ids'].shape[0] == 3, "Test 5a failed: batch size wrong"
    assert 'attention_mask' in batch, "Test 5b failed: missing attention_mask"

    # Test 6: Chunk long text
    long_text = " ".join(["word"] * 100)
    chunks = chunk_long_text(long_text, tokenizer, chunk_size=20, overlap=5)
    assert len(chunks) > 1, "Test 6a failed: should create multiple chunks"
    assert all(isinstance(c, str) for c in chunks), "Test 6b failed: chunks should be strings"

    # Test 7: Prepare recipe dataset
    prepared = prepare_recipe_dataset(SAMPLE_RECIPES, tokenizer)
    assert len(prepared) == 2, f"Test 7a failed: expected 2, got {len(prepared)}"
    assert all('input_ids' in ex for ex in prepared), "Test 7b failed: missing input_ids"

    # Test 8: Filter by length
    examples = [
        {'input_ids': [1, 2, 3]},
        {'input_ids': list(range(100))},
        {'input_ids': list(range(1000))}
    ]
    filtered = filter_by_length(examples, min_length=5, max_length=500)
    assert len(filtered) == 1, f"Test 8 failed: expected 1 after filtering, got {len(filtered)}"

    # Test 9: Dataset statistics
    examples = [
        {'input_ids': [1, 2, 3, 4, 5]},
        {'input_ids': [1, 2, 3]},
        {'input_ids': [1, 2, 3, 4, 5, 6, 7]}
    ]
    stats = dataset_statistics(examples)
    assert stats['num_examples'] == 3, "Test 9a failed: wrong count"
    assert stats['avg_length'] == 5, f"Test 9b failed: expected avg 5, got {stats['avg_length']}"
    assert stats['total_tokens'] == 15, f"Test 9c failed: expected 15, got {stats['total_tokens']}"

    # Test 10: 2D labels
    input_ids_2d = np.array([[2, 5, 6, 0], [2, 7, 8, 9]])
    labels_2d = create_labels(input_ids_2d, pad_token_id=0)
    assert labels_2d.shape == input_ids_2d.shape, "Test 10a failed: shape mismatch"
    assert labels_2d[0, -1] == -100, "Test 10b failed: padding not masked"
    assert labels_2d[1, -1] != -100, "Test 10c failed: non-padding incorrectly masked"

    print("All tests passed!")
    print("\nProper tokenization is critical for training quality.")
    print("Remember: labels = shifted input_ids, padding = -100")
