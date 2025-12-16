# Problem 8: Embedding Lookup
#
# Implement an embedding lookup table.
# An embedding table maps discrete IDs to dense vectors.
#
# Example:
#   embedding = EmbeddingTable(vocab_size=3, embed_dim=4)
#   # Creates a 3x4 matrix of random weights
#   
#   embedding.lookup([0, 2])  
#   # Returns 2x4 matrix: rows 0 and 2 from the weight matrix
#
# Constraints:
#   - Initialize weights randomly using np.random.randn
#   - lookup() should handle a list of IDs
#   - Use NumPy indexing (no loops)
#
# ML Relevance: This is exactly what nn.Embedding does in PyTorch.
# Words/tokens are converted to IDs, then looked up in embedding table.

import numpy as np


class EmbeddingTable:
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        # Your solution here
        # Use np.random.seed(seed) for reproducibility in tests
        pass
    
    def lookup(self, ids: list[int]) -> np.ndarray:
        """Look up embeddings for given IDs."""
        # Your solution here
        pass
    
    @property
    def shape(self) -> tuple:
        """Return shape of embedding table (vocab_size, embed_dim)."""
        # Your solution here
        pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Shape is correct
    emb = EmbeddingTable(vocab_size=100, embed_dim=64)
    assert emb.shape == (100, 64), f"Test 1 failed: expected (100, 64), got {emb.shape}"
    
    # Test 2: Single lookup
    emb = EmbeddingTable(vocab_size=10, embed_dim=4, seed=42)
    result = emb.lookup([0])
    assert result.shape == (1, 4), f"Test 2 failed: expected (1, 4), got {result.shape}"
    
    # Test 3: Multiple lookups
    result = emb.lookup([0, 1, 2])
    assert result.shape == (3, 4), f"Test 3 failed: expected (3, 4), got {result.shape}"
    
    # Test 4: Same ID returns same vector
    emb = EmbeddingTable(vocab_size=10, embed_dim=4, seed=42)
    v1 = emb.lookup([5])
    v2 = emb.lookup([5])
    assert np.allclose(v1, v2), "Test 4 failed: same ID should return same vector"
    
    # Test 5: Different IDs return different vectors
    result = emb.lookup([0, 1])
    assert not np.allclose(result[0], result[1]), "Test 5 failed: different IDs should have different vectors"
    
    print("All tests passed!")
