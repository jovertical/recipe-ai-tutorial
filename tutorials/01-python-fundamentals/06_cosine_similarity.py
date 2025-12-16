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
    # https://www.geeksforgeeks.org/dbms/cosine-similarity/

    # Given that x `{1, 2, 3}` and y `{ 2, 1, 4 }`

    # 1. Calculate the product (dot):
    #      1. We multiply adjacent vectors in NumP (a * b):
    #      2. Sum all the vector product like so 1*2 + 2*1 + 3*4 = 13
    dot_product = np.sum(a * b)

    # 2. Calculate the magnitude of "A"
    #      1. Calculate the dot product of "A": 1*1 + 2*2 + 3*3 = 14
    #      2. Get the square root of the calculated dot product of "A": np.sqrt(14) = 3.7416573867739413
    magnitude_of_a = np.sqrt(np.dot(a, a)) # Or using the Linear algebra functions: `np.linalg.norm(a)`\

    # 3. Calculate the magnitude of "B"
    #      1. Calculate the dot product of "B": 2*2 + 1*1 + 4*4 = 21
    #      2. Get the square root of the calculated dot product of "B": np.sqrt(21) = 4.58257569495584
    magnitude_of_b = np.sqrt(np.dot(b, b)) # Or using the Linear algebra functions: `np.linalg.norm(b)`

    # 4. Ensure that if either magnitudes of "A" or "B" does not contain zeros or else return 0.0
    #    This way, we prevent "division by zero"
    if magnitude_of_a == 0 or magnitude_of_b == 0:
        return 0.0

    # 5. Let's calculate the regular product of A and B (||a|| x ||b||)
    return dot_product / (magnitude_of_a * magnitude_of_b)

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
