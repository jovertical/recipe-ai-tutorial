"""
Exercise 15: Data Collator for Language Modeling

In this exercise, you'll implement a data collator that properly batches
tokenized sequences for language model training. The collator handles:
1. Dynamic padding to the longest sequence in a batch
2. Creating attention masks for padded positions
3. Setting up labels for causal language modeling

Example:
    # Two sequences of different lengths
    batch = [
        {"input_ids": [1, 2, 3, 4]},
        {"input_ids": [5, 6]}
    ]
    
    # After collation with pad_token_id=0:
    result = {
        "input_ids": [[1, 2, 3, 4], [5, 6, 0, 0]],
        "attention_mask": [[1, 1, 1, 1], [1, 1, 0, 0]],
        "labels": [[1, 2, 3, 4], [5, 6, -100, -100]]
    }
    
    Note: Labels use -100 for padding (PyTorch cross-entropy ignores -100)

ML Relevance:
    Data collation is crucial for efficient training. Without proper batching,
    you'd have to train one sequence at a time. The collator ensures:
    - All sequences in a batch have the same length (required for tensors)
    - Padded positions don't contribute to the loss (via -100 in labels)
    - Attention masks prevent the model from attending to padding

Your Task:
    1. Implement pad_sequences() - pad a list of sequences to the same length
    2. Implement create_attention_mask() - mask out padding positions
    3. Implement create_labels() - set up labels for causal LM (ignore padding)
    4. Implement DataCollatorForCausalLM class - complete collator

Run:
    python tutorials/06-fine-tuning/15_data_collator.py
"""

from typing import List, Dict, Any
import numpy as np


def pad_sequences(
    sequences: List[List[int]],
    pad_value: int = 0,
    padding_side: str = "right"
) -> np.ndarray:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of sequences (each is a list of integers)
        pad_value: Value to use for padding (usually 0)
        padding_side: "right" pads at end, "left" pads at beginning
        
    Returns:
        2D numpy array of shape (batch_size, max_length)
        
    Example:
        >>> seqs = [[1, 2, 3], [4, 5]]
        >>> pad_sequences(seqs, pad_value=0)
        array([[1, 2, 3],
               [4, 5, 0]])
        >>> pad_sequences(seqs, pad_value=0, padding_side="left")
        array([[1, 2, 3],
               [0, 4, 5]])
    """
    # Your solution here
    # Hints:
    # 1. Find the max length across all sequences
    # 2. Create output array filled with pad_value
    # 3. Copy each sequence to the right position based on padding_side
    pass


def create_attention_mask(
    input_ids: np.ndarray,
    pad_token_id: int = 0
) -> np.ndarray:
    """
    Create attention mask (1 for real tokens, 0 for padding).
    
    Args:
        input_ids: 2D array of token IDs (batch_size, seq_length)
        pad_token_id: Token ID used for padding
        
    Returns:
        2D array of same shape with 1s for real tokens, 0s for padding
        
    Example:
        >>> ids = np.array([[1, 2, 3, 0], [4, 5, 0, 0]])
        >>> create_attention_mask(ids, pad_token_id=0)
        array([[1, 1, 1, 0],
               [1, 1, 0, 0]])
    """
    # Your solution here
    # Hint: Compare input_ids to pad_token_id
    pass


def create_labels(
    input_ids: np.ndarray,
    pad_token_id: int = 0,
    ignore_index: int = -100
) -> np.ndarray:
    """
    Create labels for causal language modeling.
    
    For causal LM, labels are the same as input_ids, but padding positions
    are set to ignore_index (-100) so they don't contribute to the loss.
    
    Args:
        input_ids: 2D array of token IDs
        pad_token_id: Token ID used for padding
        ignore_index: Value to use for ignored positions (PyTorch uses -100)
        
    Returns:
        2D array of labels
        
    Example:
        >>> ids = np.array([[1, 2, 3, 0], [4, 5, 0, 0]])
        >>> create_labels(ids, pad_token_id=0)
        array([[   1,    2,    3, -100],
               [   4,    5, -100, -100]])
    """
    # Your solution here
    # Hints:
    # 1. Copy input_ids
    # 2. Replace pad positions with ignore_index
    pass


class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling.
    
    This class is callable - it takes a batch of examples and returns
    a dictionary with properly batched tensors.
    
    Example:
        collator = DataCollatorForCausalLM(pad_token_id=0)
        batch = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5, 6, 7]}
        ]
        result = collator(batch)
        # result["input_ids"] shape: (2, 4)
        # result["attention_mask"] shape: (2, 4)
        # result["labels"] shape: (2, 4)
    """
    
    def __init__(
        self,
        pad_token_id: int = 0,
        padding_side: str = "right",
        max_length: int = None
    ):
        """
        Initialize the data collator.
        
        Args:
            pad_token_id: Token ID to use for padding
            padding_side: "right" or "left" padding
            max_length: Optional max length to truncate to
        """
        # Your solution here
        pass
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Collate a batch of examples.
        
        Args:
            batch: List of dicts, each with "input_ids" key
            
        Returns:
            Dictionary with:
            - input_ids: padded input IDs (batch_size, seq_length)
            - attention_mask: attention mask (batch_size, seq_length)
            - labels: labels for LM loss (batch_size, seq_length)
        """
        # Your solution here
        # Hints:
        # 1. Extract input_ids from each example
        # 2. Optionally truncate to max_length
        # 3. Use pad_sequences, create_attention_mask, create_labels
        pass


def collate_with_truncation(
    sequences: List[List[int]],
    max_length: int,
    truncation_side: str = "right"
) -> List[List[int]]:
    """
    Truncate sequences that exceed max_length.
    
    Args:
        sequences: List of sequences
        max_length: Maximum allowed length
        truncation_side: "right" truncates end, "left" truncates beginning
        
    Returns:
        List of truncated sequences
        
    Example:
        >>> seqs = [[1, 2, 3, 4, 5], [6, 7]]
        >>> collate_with_truncation(seqs, max_length=3, truncation_side="right")
        [[1, 2, 3], [6, 7]]
        >>> collate_with_truncation(seqs, max_length=3, truncation_side="left")
        [[3, 4, 5], [6, 7]]
    """
    # Your solution here
    pass


def dynamic_batch_by_length(
    examples: List[Dict[str, Any]],
    max_tokens: int = 1024
) -> List[List[Dict[str, Any]]]:
    """
    Create batches that have approximately the same total tokens.
    
    This is more efficient than fixed batch sizes because short sequences
    can be batched together in larger groups.
    
    Args:
        examples: List of examples with "input_ids" key
        max_tokens: Maximum total tokens per batch
        
    Returns:
        List of batches, where each batch is a list of examples
        
    Example:
        >>> examples = [
        ...     {"input_ids": [1, 2, 3]},        # length 3
        ...     {"input_ids": [4, 5, 6, 7, 8]},  # length 5  
        ...     {"input_ids": [9, 10]},          # length 2
        ...     {"input_ids": [11, 12, 13]}      # length 3
        ... ]
        >>> batches = dynamic_batch_by_length(examples, max_tokens=8)
        # Could return: [[ex1, ex3, ex4], [ex2]] or similar groupings
    """
    # Your solution here
    # Hints:
    # 1. Sort examples by length for better packing
    # 2. Greedily add examples to batch until max_tokens exceeded
    # 3. Start new batch when current is full
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    print("Testing pad_sequences...")
    
    # Test 1: Basic right padding
    seqs = [[1, 2, 3], [4, 5], [6]]
    result = pad_sequences(seqs, pad_value=0)
    assert result.shape == (3, 3), f"Test 1a failed: wrong shape {result.shape}"
    assert np.array_equal(result[0], [1, 2, 3]), "Test 1b failed: first seq wrong"
    assert np.array_equal(result[1], [4, 5, 0]), "Test 1c failed: second seq wrong"
    assert np.array_equal(result[2], [6, 0, 0]), "Test 1d failed: third seq wrong"
    print("  ✓ Right padding works")
    
    # Test 2: Left padding
    result = pad_sequences(seqs, pad_value=0, padding_side="left")
    assert np.array_equal(result[1], [0, 4, 5]), "Test 2a failed: left padding wrong"
    assert np.array_equal(result[2], [0, 0, 6]), "Test 2b failed: left padding wrong"
    print("  ✓ Left padding works")
    
    # Test 3: Custom pad value
    result = pad_sequences([[1, 2], [3]], pad_value=99)
    assert result[1, 1] == 99, "Test 3 failed: custom pad value not used"
    print("  ✓ Custom pad value works")
    
    print("\nTesting create_attention_mask...")
    
    # Test 4: Attention mask
    ids = np.array([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = create_attention_mask(ids, pad_token_id=0)
    assert np.array_equal(mask[0], [1, 1, 1, 0]), "Test 4a failed"
    assert np.array_equal(mask[1], [1, 1, 0, 0]), "Test 4b failed"
    print("  ✓ Attention mask correct")
    
    # Test 5: Different pad token
    ids = np.array([[1, 0, 2, 50256], [3, 50256, 50256, 50256]])
    mask = create_attention_mask(ids, pad_token_id=50256)
    assert np.array_equal(mask[0], [1, 1, 1, 0]), "Test 5a failed"
    assert np.array_equal(mask[1], [1, 0, 0, 0]), "Test 5b failed"
    print("  ✓ Custom pad token ID works")
    
    print("\nTesting create_labels...")
    
    # Test 6: Labels with ignore index
    ids = np.array([[1, 2, 3, 0], [4, 5, 0, 0]])
    labels = create_labels(ids, pad_token_id=0)
    assert np.array_equal(labels[0], [1, 2, 3, -100]), "Test 6a failed"
    assert np.array_equal(labels[1], [4, 5, -100, -100]), "Test 6b failed"
    print("  ✓ Labels correct")
    
    print("\nTesting DataCollatorForCausalLM...")
    
    # Test 7: Full collator
    collator = DataCollatorForCausalLM(pad_token_id=0)
    batch = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5, 6, 7]}
    ]
    result = collator(batch)
    
    assert "input_ids" in result, "Test 7a failed: missing input_ids"
    assert "attention_mask" in result, "Test 7b failed: missing attention_mask"
    assert "labels" in result, "Test 7c failed: missing labels"
    assert result["input_ids"].shape == (2, 4), f"Test 7d failed: wrong shape {result['input_ids'].shape}"
    assert np.array_equal(result["input_ids"][0], [1, 2, 3, 0]), "Test 7e failed"
    assert np.array_equal(result["attention_mask"][1], [1, 1, 1, 1]), "Test 7f failed"
    assert result["labels"][0, 3] == -100, "Test 7g failed: padding not ignored"
    print("  ✓ DataCollator works correctly")
    
    # Test 8: With max_length truncation
    collator = DataCollatorForCausalLM(pad_token_id=0, max_length=3)
    batch = [
        {"input_ids": [1, 2, 3, 4, 5]},
        {"input_ids": [6, 7]}
    ]
    result = collator(batch)
    assert result["input_ids"].shape == (2, 3), f"Test 8a failed: wrong shape {result['input_ids'].shape}"
    print("  ✓ Max length truncation works")
    
    print("\nTesting collate_with_truncation...")
    
    # Test 9: Right truncation
    seqs = [[1, 2, 3, 4, 5], [6, 7]]
    result = collate_with_truncation(seqs, max_length=3, truncation_side="right")
    assert result[0] == [1, 2, 3], f"Test 9a failed: {result[0]}"
    assert result[1] == [6, 7], f"Test 9b failed: {result[1]}"
    print("  ✓ Right truncation works")
    
    # Test 10: Left truncation
    result = collate_with_truncation(seqs, max_length=3, truncation_side="left")
    assert result[0] == [3, 4, 5], f"Test 10 failed: {result[0]}"
    print("  ✓ Left truncation works")
    
    print("\nTesting dynamic_batch_by_length...")
    
    # Test 11: Dynamic batching
    examples = [
        {"input_ids": [1, 2, 3]},        # length 3
        {"input_ids": [4, 5, 6, 7, 8]},  # length 5
        {"input_ids": [9, 10]},          # length 2
        {"input_ids": [11, 12, 13]}      # length 3
    ]
    batches = dynamic_batch_by_length(examples, max_tokens=8)
    
    # Check that all examples are in some batch
    all_examples = []
    for batch in batches:
        all_examples.extend(batch)
    assert len(all_examples) == 4, f"Test 11a failed: not all examples in batches"
    
    # Check that no batch exceeds max_tokens (approximately)
    for batch in batches:
        total = sum(len(ex["input_ids"]) for ex in batch)
        # Allow some slack for last example that pushed over limit
        assert total <= 13, f"Test 11b failed: batch too large ({total} tokens)"
    print("  ✓ Dynamic batching works")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    print("\nExcellent! You've implemented a data collator for causal LM training.")
    print("Key takeaways:")
    print("- Padding allows batching sequences of different lengths")
    print("- Attention masks prevent attending to padding tokens")
    print("- Labels use -100 so padding doesn't affect the loss")
    print("- Dynamic batching improves training efficiency")
