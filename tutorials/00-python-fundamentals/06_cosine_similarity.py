# Problem 6: Cosine Similarity
#
# Implement cosine similarity between two vectors using NumPy.
# Cosine similarity measures how similar two vectors are by the angle between them.
#
# Formula: cos(a, b) = (a . b) / (||a|| * ||b||)
# Where:
#   - a . b is the dot product
#   - ||a|| is the magnitude (L2 norm) of a
#
# Example:
#   a = [1, 0, 0]
#   b = [1, 0, 0]
#   cosine_similarity(a, b) = 1.0  (identical direction)
#
#   a = [1, 0, 0]
#   b = [0, 1, 0]
#   cosine_similarity(a, b) = 0.0  (perpendicular)
#
# Constraints:
#   - Use NumPy (no loops)
#   - Handle zero vectors by returning 0.0
#   - Return a float
#
# ML Relevance: Embeddings are compared using cosine similarity.
# "How similar is ingredient A to ingredient B?" = cosine similarity of their embeddings.

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Identical vectors
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    result = cosine_similarity(a, b)
    assert abs(result - 1.0) < 1e-6, f"Test 1 failed: expected 1.0, got {result}"
    
    # Test 2: Opposite vectors
    a = np.array([1, 0, 0])
    b = np.array([-1, 0, 0])
    result = cosine_similarity(a, b)
    assert abs(result - (-1.0)) < 1e-6, f"Test 2 failed: expected -1.0, got {result}"
    
    # Test 3: Perpendicular vectors
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    result = cosine_similarity(a, b)
    assert abs(result - 0.0) < 1e-6, f"Test 3 failed: expected 0.0, got {result}"
    
    # Test 4: Arbitrary vectors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = cosine_similarity(a, b)
    expected = 0.9746318461970762
    assert abs(result - expected) < 1e-6, f"Test 4 failed: expected {expected}, got {result}"
    
    # Test 5: Zero vector handling
    a = np.array([0, 0, 0])
    b = np.array([1, 2, 3])
    result = cosine_similarity(a, b)
    assert result == 0.0, f"Test 5 failed: expected 0.0, got {result}"
    
    print("All tests passed!")
