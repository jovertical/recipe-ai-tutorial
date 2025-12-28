"""
Exercise 02: Dense Embeddings

In this exercise, you'll implement a simple embedding layer from scratch.
An embedding layer is just a learned lookup table: each word gets a dense
vector that is learned during training.

Example:
    # Vocabulary: {"apple": 0, "banana": 1, "cherry": 2}
    # Embedding dimension: 4
    
    embedding_matrix = [
        [0.2, -0.1, 0.5, 0.3],   # apple
        [-0.4, 0.6, 0.1, -0.2],  # banana  
        [0.1, 0.3, -0.5, 0.4]    # cherry
    ]
    
    embed("apple") = [0.2, -0.1, 0.5, 0.3]  # Just lookup row 0

ML Relevance:
    Embeddings are fundamental to NLP:
    - Dense vectors (50-1000 dims) instead of sparse one-hot (50,000+ dims)
    - Similar words have similar vectors
    - Learned end-to-end with the model
    - Transfer learning: pretrained embeddings capture world knowledge

Your Task:
    1. Implement EmbeddingLayer class with random initialization
    2. Implement forward() - look up embeddings by index
    3. Implement embed_words() - embed a list of words
    4. Understand how gradients flow through embeddings

Run:
    python tutorials/07-embeddings/02_dense_embeddings.py
"""

from typing import List, Dict, Union
import numpy as np


class EmbeddingLayer:
    """
    A simple embedding layer (lookup table).
    
    Attributes:
        num_embeddings: Size of vocabulary
        embedding_dim: Dimension of each embedding vector
        weight: The embedding matrix (num_embeddings, embedding_dim)
    
    Example:
        >>> emb = EmbeddingLayer(num_embeddings=1000, embedding_dim=64)
        >>> emb.weight.shape
        (1000, 64)
        >>> emb(np.array([0, 5, 10])).shape
        (3, 64)
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None
    ):
        """
        Initialize the embedding layer.
        
        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of embedding vectors
            padding_idx: If set, embeddings at this index are zeros
        """
        # Your solution here
        # Hints:
        # 1. Store the parameters
        # 2. Initialize weight matrix with small random values
        #    Use np.random.randn() * 0.02 (typical initialization)
        # 3. If padding_idx is set, set that row to zeros
        pass
    
    def __call__(self, indices: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for given indices.
        
        Args:
            indices: Integer array of any shape
            
        Returns:
            Embeddings with shape (*indices.shape, embedding_dim)
            
        Example:
            >>> emb = EmbeddingLayer(100, 64)
            >>> emb(np.array([1, 2, 3])).shape
            (3, 64)
            >>> emb(np.array([[1, 2], [3, 4]])).shape
            (2, 2, 64)
        """
        # Your solution here
        # Hint: NumPy fancy indexing makes this one line!
        pass
    
    def embed_words(
        self,
        words: List[str],
        vocab: Dict[str, int],
        handle_oov: str = "zero"
    ) -> np.ndarray:
        """
        Embed a list of words using vocabulary.
        
        Args:
            words: List of words to embed
            vocab: Word to index mapping
            handle_oov: How to handle out-of-vocabulary words
                       "zero" -> return zero vector
                       "unk" -> use [UNK] token (must be in vocab)
                       "random" -> return random vector
                       
        Returns:
            Embeddings of shape (len(words), embedding_dim)
            
        Example:
            >>> vocab = {"[UNK]": 0, "hello": 1, "world": 2}
            >>> emb = EmbeddingLayer(3, 64)
            >>> emb.embed_words(["hello", "unknown"], vocab, handle_oov="unk")
            # Returns embeddings for indices [1, 0]
        """
        # Your solution here
        pass


def initialize_embeddings(
    num_embeddings: int,
    embedding_dim: int,
    init_type: str = "normal"
) -> np.ndarray:
    """
    Initialize embedding weights with different strategies.
    
    Args:
        num_embeddings: Vocabulary size
        embedding_dim: Embedding dimension
        init_type: Initialization type
            "normal" -> N(0, 0.02)
            "uniform" -> U(-0.1, 0.1)
            "xavier" -> Xavier/Glorot initialization
            
    Returns:
        Initialized embedding matrix
        
    Example:
        >>> weights = initialize_embeddings(1000, 64, "xavier")
        >>> weights.shape
        (1000, 64)
        >>> abs(weights.mean()) < 0.01  # Should be centered
        True
    """
    # Your solution here
    # Hints:
    # - normal: np.random.randn() * 0.02
    # - uniform: np.random.uniform(-0.1, 0.1)
    # - xavier: np.random.randn() * sqrt(2 / (fan_in + fan_out))
    pass


def embedding_similarity(
    word1: str,
    word2: str,
    embedding_layer: EmbeddingLayer,
    vocab: Dict[str, int]
) -> float:
    """
    Calculate cosine similarity between word embeddings.
    
    Args:
        word1: First word
        word2: Second word
        embedding_layer: The embedding layer
        vocab: Word to index mapping
        
    Returns:
        Cosine similarity between -1 and 1
        
    Example:
        >>> sim = embedding_similarity("king", "queen", emb, vocab)
        >>> sim > 0.5  # Should be high for related words (after training)
        True
    """
    # Your solution here
    pass


def find_nearest_neighbors(
    word: str,
    embedding_layer: EmbeddingLayer,
    vocab: Dict[str, int],
    top_k: int = 5
) -> List[tuple]:
    """
    Find most similar words to a given word.
    
    Args:
        word: Query word
        embedding_layer: The embedding layer
        vocab: Word to index mapping
        top_k: Number of neighbors to return
        
    Returns:
        List of (word, similarity) tuples, sorted by similarity
        
    Example:
        >>> neighbors = find_nearest_neighbors("king", emb, vocab, top_k=5)
        >>> neighbors[0]
        ('king', 1.0)  # Most similar is itself
        >>> neighbors[1][0]  # Second most similar
        'queen'  # (after training on appropriate data)
    """
    # Your solution here
    # Hints:
    # 1. Get embedding for query word
    # 2. Compute similarity with all embeddings
    # 3. Sort and return top k
    pass


def compute_embedding_statistics(
    embedding_layer: EmbeddingLayer
) -> Dict[str, float]:
    """
    Compute statistics about embedding weights.
    
    Args:
        embedding_layer: The embedding layer
        
    Returns:
        Dictionary with statistics:
        - mean: Mean of all weights
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - norm_mean: Mean L2 norm of embeddings
        
    Example:
        >>> stats = compute_embedding_statistics(emb)
        >>> abs(stats["mean"]) < 0.1  # Should be near zero
        True
    """
    # Your solution here
    pass


def average_embeddings(
    words: List[str],
    embedding_layer: EmbeddingLayer,
    vocab: Dict[str, int]
) -> np.ndarray:
    """
    Compute average embedding for a list of words.
    
    This is a simple way to get sentence/phrase embeddings.
    
    Args:
        words: List of words
        embedding_layer: The embedding layer
        vocab: Word to index mapping
        
    Returns:
        Average embedding vector
        
    Example:
        >>> avg = average_embeddings(["hello", "world"], emb, vocab)
        >>> avg.shape
        (64,)  # Same as embedding_dim
    """
    # Your solution here
    pass


class EmbeddingWithDropout(EmbeddingLayer):
    """
    Embedding layer with dropout for regularization.
    
    Dropout randomly zeros out entire embeddings during training.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dropout_prob: float = 0.1,
        padding_idx: int = None
    ):
        """
        Initialize embedding with dropout.
        
        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Embedding dimension
            dropout_prob: Probability of dropping an embedding
            padding_idx: Index for padding token
        """
        # Your solution here
        # Hint: Call parent __init__ and store dropout_prob
        pass
    
    def __call__(
        self,
        indices: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Look up embeddings with optional dropout.
        
        Args:
            indices: Integer indices to look up
            training: Whether in training mode (apply dropout)
            
        Returns:
            Embeddings, possibly with some dropped to zero
        """
        # Your solution here
        # Hints:
        # 1. Get base embeddings
        # 2. If training, randomly zero some out
        # 3. Scale remaining by 1/(1-p) to maintain expected value
        pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing EmbeddingLayer initialization...")
    
    # Test 1: Basic initialization
    emb = EmbeddingLayer(num_embeddings=100, embedding_dim=64)
    assert emb.weight.shape == (100, 64), f"Test 1a failed: {emb.weight.shape}"
    assert abs(emb.weight.mean()) < 0.1, "Test 1b failed: mean should be near 0"
    print("  ✓ Basic initialization works")
    
    # Test 2: Padding index
    emb = EmbeddingLayer(num_embeddings=100, embedding_dim=64, padding_idx=0)
    assert np.allclose(emb.weight[0], 0), "Test 2 failed: padding should be zeros"
    print("  ✓ Padding index works")
    
    print("\nTesting EmbeddingLayer forward...")
    
    # Test 3: Single lookup
    emb = EmbeddingLayer(num_embeddings=100, embedding_dim=64)
    result = emb(np.array([5]))
    assert result.shape == (1, 64), f"Test 3a failed: {result.shape}"
    assert np.allclose(result[0], emb.weight[5]), "Test 3b failed: wrong embedding"
    print("  ✓ Single lookup works")
    
    # Test 4: Batch lookup
    result = emb(np.array([1, 5, 10, 20]))
    assert result.shape == (4, 64), f"Test 4a failed: {result.shape}"
    for i, idx in enumerate([1, 5, 10, 20]):
        assert np.allclose(result[i], emb.weight[idx]), f"Test 4b failed at {i}"
    print("  ✓ Batch lookup works")
    
    # Test 5: 2D indices
    result = emb(np.array([[1, 2], [3, 4]]))
    assert result.shape == (2, 2, 64), f"Test 5 failed: {result.shape}"
    print("  ✓ 2D indices work")
    
    print("\nTesting embed_words...")
    
    # Test 6: Basic word embedding
    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    emb = EmbeddingLayer(num_embeddings=3, embedding_dim=32)
    result = emb.embed_words(["hello", "world"], vocab)
    assert result.shape == (2, 32), f"Test 6 failed: {result.shape}"
    print("  ✓ Word embedding works")
    
    # Test 7: OOV handling - unk
    result = emb.embed_words(["hello", "unknown"], vocab, handle_oov="unk")
    assert np.allclose(result[1], emb.weight[0]), "Test 7 failed: OOV should use [UNK]"
    print("  ✓ OOV with [UNK] works")
    
    # Test 8: OOV handling - zero
    result = emb.embed_words(["unknown"], vocab, handle_oov="zero")
    assert np.allclose(result[0], 0), "Test 8 failed: OOV should be zeros"
    print("  ✓ OOV with zero works")
    
    print("\nTesting initialize_embeddings...")
    
    # Test 9: Normal initialization
    weights = initialize_embeddings(1000, 64, "normal")
    assert weights.shape == (1000, 64), f"Test 9a failed: {weights.shape}"
    assert abs(weights.mean()) < 0.01, f"Test 9b failed: mean {weights.mean()}"
    print("  ✓ Normal initialization works")
    
    # Test 10: Xavier initialization
    weights = initialize_embeddings(1000, 64, "xavier")
    expected_std = np.sqrt(2 / (1000 + 64))
    actual_std = weights.std()
    assert abs(actual_std - expected_std) < 0.01, f"Test 10 failed: std {actual_std} vs {expected_std}"
    print("  ✓ Xavier initialization works")
    
    print("\nTesting embedding_similarity...")
    
    # Test 11: Self similarity
    vocab = {"word1": 0, "word2": 1, "word3": 2}
    emb = EmbeddingLayer(num_embeddings=3, embedding_dim=32)
    sim = embedding_similarity("word1", "word1", emb, vocab)
    assert abs(sim - 1.0) < 0.001, f"Test 11 failed: self-similarity should be 1, got {sim}"
    print("  ✓ Self-similarity = 1.0")
    
    # Test 12: Different words
    sim = embedding_similarity("word1", "word2", emb, vocab)
    assert -1 <= sim <= 1, f"Test 12 failed: similarity out of range {sim}"
    print("  ✓ Similarity in valid range")
    
    print("\nTesting find_nearest_neighbors...")
    
    # Test 13: Find neighbors
    vocab = {f"word{i}": i for i in range(10)}
    emb = EmbeddingLayer(num_embeddings=10, embedding_dim=32)
    neighbors = find_nearest_neighbors("word0", emb, vocab, top_k=3)
    assert len(neighbors) == 3, f"Test 13a failed: {len(neighbors)}"
    assert neighbors[0][0] == "word0", f"Test 13b failed: first should be query"
    assert neighbors[0][1] == 1.0, f"Test 13c failed: self-similarity should be 1"
    print("  ✓ Nearest neighbors found")
    
    print("\nTesting compute_embedding_statistics...")
    
    # Test 14: Statistics
    emb = EmbeddingLayer(num_embeddings=100, embedding_dim=64)
    stats = compute_embedding_statistics(emb)
    assert "mean" in stats, "Test 14a failed"
    assert "std" in stats, "Test 14b failed"
    assert "norm_mean" in stats, "Test 14c failed"
    assert stats["norm_mean"] > 0, "Test 14d failed"
    print("  ✓ Statistics computed")
    
    print("\nTesting average_embeddings...")
    
    # Test 15: Average embedding
    vocab = {"hello": 0, "world": 1, "test": 2}
    emb = EmbeddingLayer(num_embeddings=3, embedding_dim=32)
    avg = average_embeddings(["hello", "world"], emb, vocab)
    assert avg.shape == (32,), f"Test 15a failed: {avg.shape}"
    expected = (emb.weight[0] + emb.weight[1]) / 2
    assert np.allclose(avg, expected), "Test 15b failed: average wrong"
    print("  ✓ Average embedding works")
    
    print("\nTesting EmbeddingWithDropout...")
    
    # Test 16: Dropout initialization
    emb_drop = EmbeddingWithDropout(100, 64, dropout_prob=0.1)
    assert emb_drop.weight.shape == (100, 64), "Test 16 failed"
    print("  ✓ Dropout embedding initialized")
    
    # Test 17: Dropout in training
    np.random.seed(42)
    result_train = emb_drop(np.array([1, 2, 3, 4, 5]), training=True)
    
    # Test 18: No dropout in eval
    result_eval = emb_drop(np.array([1, 2, 3, 4, 5]), training=False)
    # In eval mode, should match base embeddings
    for i, idx in enumerate([1, 2, 3, 4, 5]):
        assert np.allclose(result_eval[i], emb_drop.weight[idx]), f"Test 18 failed at {i}"
    print("  ✓ Dropout respects training mode")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    print("\nKey takeaways:")
    print("- Embeddings are just learned lookup tables")
    print("- Each word gets a dense vector (vs sparse one-hot)")
    print("- Similarity can be computed with cosine similarity")
    print("- Dropout helps regularization during training")
