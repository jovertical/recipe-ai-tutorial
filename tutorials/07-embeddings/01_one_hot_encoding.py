"""
Exercise 01: One-Hot Encoding

In this exercise, you'll implement one-hot encoding from scratch to understand
what embeddings replace. One-hot encoding represents each word as a sparse
vector with a single 1 and all other positions as 0.

Example:
    vocabulary = ["apple", "banana", "cherry"]
    
    one_hot("apple")  = [1, 0, 0]
    one_hot("banana") = [0, 1, 0]
    one_hot("cherry") = [0, 0, 1]

ML Relevance:
    One-hot encoding was the standard before embeddings. It has problems:
    - High dimensionality: vocab_size can be 50,000+
    - No semantic similarity: similar words are equally distant
    - Sparse vectors: wasteful memory and computation
    
    Embeddings solve these by learning dense, semantic representations.

Your Task:
    1. Implement create_vocabulary() - build word to index mapping
    2. Implement one_hot_encode() - encode a single word
    3. Implement one_hot_encode_sequence() - encode multiple words
    4. Implement one_hot_similarity() - show why one-hot fails for similarity

Run:
    python tutorials/07-embeddings/01_one_hot_encoding.py
"""

from typing import List, Dict, Tuple
import numpy as np


def create_vocabulary(words: List[str], add_special_tokens: bool = True) -> Dict[str, int]:
    """
    Create a vocabulary mapping from words to indices.
    
    Args:
        words: List of words to include in vocabulary
        add_special_tokens: Whether to add [UNK] and [PAD] tokens
        
    Returns:
        Dictionary mapping word -> index
        
    Example:
        >>> vocab = create_vocabulary(["apple", "banana"])
        >>> vocab["apple"]
        2  # After [PAD]=0, [UNK]=1
        >>> vocab["banana"]
        3
    """
    # Your solution here
    # Hints:
    # 1. If add_special_tokens, start with [PAD] -> 0, [UNK] -> 1
    # 2. Add each unique word with incrementing index
    # 3. Use set() to get unique words, then sort for reproducibility
    pass


def one_hot_encode(word: str, vocabulary: Dict[str, int]) -> np.ndarray:
    """
    Create one-hot encoding for a single word.
    
    Args:
        word: Word to encode
        vocabulary: Word to index mapping
        
    Returns:
        One-hot vector of length vocab_size
        
    Example:
        >>> vocab = {"[PAD]": 0, "[UNK]": 1, "apple": 2, "banana": 3}
        >>> one_hot_encode("apple", vocab)
        array([0., 0., 1., 0.])
        >>> one_hot_encode("unknown", vocab)  # Returns UNK
        array([0., 1., 0., 0.])
    """
    # Your solution here
    # Hints:
    # 1. Create zero vector of vocab_size
    # 2. Get index for word (use [UNK] if not found)
    # 3. Set that position to 1
    pass


def one_hot_encode_sequence(
    words: List[str],
    vocabulary: Dict[str, int]
) -> np.ndarray:
    """
    Create one-hot encodings for a sequence of words.
    
    Args:
        words: List of words to encode
        vocabulary: Word to index mapping
        
    Returns:
        2D array of shape (num_words, vocab_size)
        
    Example:
        >>> vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
        >>> encodings = one_hot_encode_sequence(["hello", "world"], vocab)
        >>> encodings.shape
        (2, 4)
    """
    # Your solution here
    pass


def one_hot_similarity(
    word1: str,
    word2: str,
    vocabulary: Dict[str, int]
) -> float:
    """
    Calculate cosine similarity between one-hot encodings.
    
    This demonstrates why one-hot encoding fails for semantic similarity.
    
    Args:
        word1: First word
        word2: Second word
        vocabulary: Word to index mapping
        
    Returns:
        Cosine similarity (0 if different words, 1 if same word)
        
    Example:
        >>> vocab = create_vocabulary(["cat", "dog", "car"])
        >>> one_hot_similarity("cat", "dog", vocab)  # Similar animals
        0.0  # But one-hot sees them as completely different!
        >>> one_hot_similarity("cat", "car", vocab)  # Very different
        0.0  # Same distance as cat-dog
    """
    # Your solution here
    # Hints:
    # 1. Get one-hot encodings for both words
    # 2. Calculate cosine similarity: dot(a, b) / (norm(a) * norm(b))
    pass


def analyze_one_hot_problems(vocabulary: Dict[str, int]) -> Dict[str, any]:
    """
    Analyze problems with one-hot encoding.
    
    Args:
        vocabulary: Word to index mapping
        
    Returns:
        Dictionary with analysis:
        - vocab_size: Number of words
        - memory_per_word_bytes: Memory for one encoding (float32)
        - sparsity: Fraction of zeros in encoding
        - total_pairs: Number of word pairs
        - same_similarity_pairs: All pairs have same similarity
        
    Example:
        >>> vocab = create_vocabulary(["the", "a", "cat", "dog", "runs"])
        >>> analysis = analyze_one_hot_problems(vocab)
        >>> analysis["sparsity"] > 0.99  # Almost all zeros
        True
    """
    # Your solution here
    pass


def compare_representations(
    word_pairs: List[Tuple[str, str]],
    vocabulary: Dict[str, int],
    embeddings: Dict[str, np.ndarray] = None
) -> Dict[str, List[float]]:
    """
    Compare one-hot vs embedding similarities for word pairs.
    
    Args:
        word_pairs: List of (word1, word2) tuples
        vocabulary: Word to index mapping
        embeddings: Optional pretrained embeddings (word -> vector)
        
    Returns:
        Dictionary with:
        - one_hot_similarities: List of one-hot similarities
        - embedding_similarities: List of embedding similarities (if provided)
        
    Example:
        >>> pairs = [("cat", "dog"), ("cat", "car"), ("dog", "puppy")]
        >>> result = compare_representations(pairs, vocab, embeddings)
        >>> result["one_hot_similarities"]
        [0.0, 0.0, 0.0]  # All same!
        >>> result["embedding_similarities"]
        [0.8, 0.2, 0.9]  # Captures semantic similarity
    """
    # Your solution here
    pass


def create_mock_embeddings(vocabulary: Dict[str, int], dim: int = 50) -> Dict[str, np.ndarray]:
    """
    Create mock embeddings with some semantic structure.
    
    This is for testing - real embeddings come from training.
    Similar words will have similar vectors.
    
    Args:
        vocabulary: Word to index mapping
        dim: Embedding dimension
        
    Returns:
        Dictionary mapping word -> embedding vector
        
    Example:
        >>> embeddings = create_mock_embeddings(vocab, dim=50)
        >>> embeddings["cat"].shape
        (50,)
    """
    # Your solution here
    # Create random embeddings, but make similar words closer
    # For simplicity, just create random vectors
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    print("Testing create_vocabulary...")
    
    # Test 1: Basic vocabulary
    vocab = create_vocabulary(["apple", "banana", "cherry"])
    assert "[PAD]" in vocab, "Test 1a failed: missing [PAD]"
    assert "[UNK]" in vocab, "Test 1b failed: missing [UNK]"
    assert vocab["[PAD]"] == 0, "Test 1c failed: [PAD] should be 0"
    assert vocab["[UNK]"] == 1, "Test 1d failed: [UNK] should be 1"
    assert len(vocab) == 5, f"Test 1e failed: wrong size {len(vocab)}"
    print("  ✓ Vocabulary created with special tokens")
    
    # Test 2: Without special tokens
    vocab_no_special = create_vocabulary(["apple", "banana"], add_special_tokens=False)
    assert "[PAD]" not in vocab_no_special, "Test 2a failed"
    assert len(vocab_no_special) == 2, "Test 2b failed"
    print("  ✓ Vocabulary without special tokens works")
    
    # Test 3: Duplicate words
    vocab = create_vocabulary(["a", "b", "a", "b", "c"])
    assert len(vocab) == 5, f"Test 3 failed: duplicates not handled {len(vocab)}"  # PAD, UNK, a, b, c
    print("  ✓ Duplicates handled")
    
    print("\nTesting one_hot_encode...")
    
    # Test 4: Basic encoding
    vocab = {"[PAD]": 0, "[UNK]": 1, "apple": 2, "banana": 3}
    encoding = one_hot_encode("apple", vocab)
    assert encoding.shape == (4,), f"Test 4a failed: wrong shape {encoding.shape}"
    assert encoding[2] == 1, "Test 4b failed: wrong position"
    assert np.sum(encoding) == 1, "Test 4c failed: not one-hot"
    print("  ✓ Basic encoding works")
    
    # Test 5: Unknown word
    encoding = one_hot_encode("unknown_word", vocab)
    assert encoding[1] == 1, "Test 5 failed: unknown should map to [UNK]"
    print("  ✓ Unknown words handled")
    
    print("\nTesting one_hot_encode_sequence...")
    
    # Test 6: Sequence encoding
    vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    encodings = one_hot_encode_sequence(["hello", "world"], vocab)
    assert encodings.shape == (2, 4), f"Test 6a failed: {encodings.shape}"
    assert encodings[0, 2] == 1, "Test 6b failed"
    assert encodings[1, 3] == 1, "Test 6c failed"
    print("  ✓ Sequence encoding works")
    
    print("\nTesting one_hot_similarity...")
    
    # Test 7: Same word similarity
    vocab = create_vocabulary(["cat", "dog", "car"])
    sim = one_hot_similarity("cat", "cat", vocab)
    assert sim == 1.0, f"Test 7a failed: same word should be 1.0, got {sim}"
    print("  ✓ Same word similarity = 1.0")
    
    # Test 8: Different word similarity
    sim = one_hot_similarity("cat", "dog", vocab)
    assert sim == 0.0, f"Test 8 failed: different words should be 0.0, got {sim}"
    print("  ✓ Different word similarity = 0.0")
    
    # Test 9: Semantic similarity problem
    sim_cat_dog = one_hot_similarity("cat", "dog", vocab)  # Similar concepts
    sim_cat_car = one_hot_similarity("cat", "car", vocab)  # Different concepts
    assert sim_cat_dog == sim_cat_car, "Test 9 failed: one-hot can't distinguish semantic similarity"
    print("  ✓ One-hot fails to capture semantic similarity (as expected)")
    
    print("\nTesting analyze_one_hot_problems...")
    
    # Test 10: Analysis
    vocab = create_vocabulary(["the", "a", "cat", "dog", "runs", "jumps", "quickly"])
    analysis = analyze_one_hot_problems(vocab)
    assert "vocab_size" in analysis, "Test 10a failed"
    assert "sparsity" in analysis, "Test 10b failed"
    assert analysis["vocab_size"] == len(vocab), "Test 10c failed"
    assert analysis["sparsity"] > 0.8, f"Test 10d failed: sparsity should be high, got {analysis['sparsity']}"
    print("  ✓ Analysis computed")
    
    print("\nTesting compare_representations...")
    
    # Test 11: Comparison without embeddings
    pairs = [("cat", "dog"), ("cat", "car")]
    vocab = create_vocabulary(["cat", "dog", "car"])
    result = compare_representations(pairs, vocab)
    assert "one_hot_similarities" in result, "Test 11a failed"
    assert len(result["one_hot_similarities"]) == 2, "Test 11b failed"
    assert all(s == 0.0 for s in result["one_hot_similarities"]), "Test 11c failed"
    print("  ✓ Comparison works without embeddings")
    
    # Test 12: Comparison with embeddings
    mock_embeddings = create_mock_embeddings(vocab, dim=50)
    result = compare_representations(pairs, vocab, mock_embeddings)
    assert "embedding_similarities" in result, "Test 12a failed"
    assert len(result["embedding_similarities"]) == 2, "Test 12b failed"
    print("  ✓ Comparison works with embeddings")
    
    print("\nTesting create_mock_embeddings...")
    
    # Test 13: Mock embeddings
    vocab = create_vocabulary(["apple", "banana", "cherry"])
    embeddings = create_mock_embeddings(vocab, dim=50)
    assert len(embeddings) == len(vocab), f"Test 13a failed: {len(embeddings)} vs {len(vocab)}"
    for word, vec in embeddings.items():
        assert vec.shape == (50,), f"Test 13b failed for {word}: {vec.shape}"
    print("  ✓ Mock embeddings created")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    print("\nYou've learned why one-hot encoding has limitations:")
    print("1. Sparse vectors (mostly zeros)")
    print("2. High memory usage (vocab_size dimensions)")
    print("3. No semantic similarity (all different words equally distant)")
    print("\nEmbeddings solve these problems with dense, learned representations!")
