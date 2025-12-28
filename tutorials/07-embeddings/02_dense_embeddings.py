# Problem 2: Dense Embeddings
#
# Implement a simple embedding layer from scratch. An embedding layer is just
# a learned lookup table: each word gets a dense vector learned during training.
#
# You'll implement:
# 1. EmbeddingLayer class with random initialization
# 2. forward() - look up embeddings by index
# 3. embed_words() - embed a list of words
# 4. Utility functions for similarity and statistics
#
# Example:
#   # Vocabulary: {"apple": 0, "banana": 1, "cherry": 2}
#   # Embedding dimension: 4
#   embedding_matrix = [
#       [0.2, -0.1, 0.5, 0.3],   # apple
#       [-0.4, 0.6, 0.1, -0.2],  # banana
#       [0.1, 0.3, -0.5, 0.4]    # cherry
#   ]
#   embed("apple") = [0.2, -0.1, 0.5, 0.3]  # Just lookup row 0
#
# Constraints:
#   - Initialize with small random values (randn * 0.02)
#   - Support padding_idx for zero embeddings
#   - Handle out-of-vocabulary words with zero/unk/random options
#
# ML Relevance: Embeddings are fundamental to NLP: dense vectors (50-1000 dims)
# instead of sparse one-hot (50,000+ dims), similar words have similar vectors,
# learned end-to-end with the model, and enable transfer learning.

from typing import List, Dict, Union
import numpy as np


class EmbeddingLayer:
    """A simple embedding layer (lookup table)."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        """Initialize with random weights, optionally zeroing padding_idx."""
        # Your solution here
        pass
    
    def __call__(self, indices: np.ndarray) -> np.ndarray:
        """Look up embeddings for given indices."""
        # Your solution here
        pass
    
    def embed_words(self, words: List[str], vocab: Dict[str, int], handle_oov: str = "zero") -> np.ndarray:
        """Embed a list of words. handle_oov: 'zero', 'unk', or 'random'."""
        # Your solution here
        pass


def initialize_embeddings(num_embeddings: int, embedding_dim: int, init_type: str = "normal") -> np.ndarray:
    """Initialize embeddings: 'normal', 'uniform', or 'xavier'."""
    # Your solution here
    pass


def embedding_similarity(word1: str, word2: str, embedding_layer: EmbeddingLayer, vocab: Dict[str, int]) -> float:
    """Calculate cosine similarity between word embeddings."""
    # Your solution here
    pass


def find_nearest_neighbors(word: str, embedding_layer: EmbeddingLayer, vocab: Dict[str, int], top_k: int = 5) -> List[tuple]:
    """Find most similar words to a given word."""
    # Your solution here
    pass


def compute_embedding_statistics(embedding_layer: EmbeddingLayer) -> Dict[str, float]:
    """Compute statistics: mean, std, min, max, norm_mean."""
    # Your solution here
    pass


def average_embeddings(words: List[str], embedding_layer: EmbeddingLayer, vocab: Dict[str, int]) -> np.ndarray:
    """Compute average embedding for a list of words."""
    # Your solution here
    pass


class EmbeddingWithDropout(EmbeddingLayer):
    """Embedding layer with dropout for regularization."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, dropout_prob: float = 0.1, padding_idx: int = None):
        """Initialize embedding with dropout."""
        # Your solution here
        pass
    
    def __call__(self, indices: np.ndarray, training: bool = True) -> np.ndarray:
        """Look up embeddings with optional dropout during training."""
        # Your solution here
        pass


# ----- Tests (do not modify) -----
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
    for i, idx in enumerate([1, 2, 3, 4, 5]):
        assert np.allclose(result_eval[i], emb_drop.weight[idx]), f"Test 18 failed at {i}"
    print("  ✓ Dropout respects training mode")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
