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
    weights: np.ndarray

    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        # Use np.random.seed(seed) for reproducibility in tests

        # Affects the global random state of the numpy.random.* module
        np.random.seed(seed)

        # Sets the embedding table weights:
        #   - Shape: (vocab_size, embed_dim) → each row is a word's embedding vector
        #   - np.random.normal generates random values from a standard normal distribution
        #   - astype(np.float32) converts to 32-bit floats (common in ML for memory efficiency)
        #
        # Example with vocab_size=3, embed_dim=4:
        #   weights = [
        #       [0.49, -0.13,  1.20, -0.87],  # ID 0 ("hello")
        #       [0.72,  0.31, -0.45,  0.18],  # ID 1 ("world")
        #       [-0.22, 0.95,  0.11, -0.63],  # ID 2 ("there")
        #   ]
        #   Shape: (3, 4)
        #
        #   lookup([0, 2]) → returns rows 0 and 2:
        #   [
        #       [0.49, -0.13,  1.20, -0.87],  # ID 0
        #       [-0.22, 0.95,  0.11, -0.63],  # ID 2
        #   ]
        #   Shape: (2, 4)
        self.weights = np.random.normal(size=(vocab_size, embed_dim)).astype(np.float32)

        pass

    def lookup(self, ids: list[int]) -> np.ndarray:
        """Look up embeddings for given IDs."""

        return self.weights[ids]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of embedding table (vocab_size, embed_dim)."""

        return self.weights.shape


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
