# Problem 7: Softmax
#
# Implement the softmax function using NumPy.
# Softmax converts a vector of numbers into a probability distribution.
#
# Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
#
# Example:
#   Input:  [1.0, 2.0, 3.0]
#   Output: [0.09, 0.24, 0.67]  (approximately, sums to 1.0)
#
# Constraints:
#   - Use NumPy (no loops)
#   - Handle numerical stability (subtract max before exp)
#   - Output should sum to 1.0
#
# ML Relevance: Softmax is the final layer of classification models.
# It turns raw scores (logits) into probabilities.

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    # Your solution here
    # Hint: For numerical stability, subtract max(x) before computing exp
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Output sums to 1
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert abs(result.sum() - 1.0) < 1e-6, f"Test 1 failed: sum is {result.sum()}, expected 1.0"
    
    # Test 2: Higher values get higher probabilities
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert result[2] > result[1] > result[0], "Test 2 failed: order should be preserved"
    
    # Test 3: Known values
    x = np.array([0.0, 0.0, 0.0])
    result = softmax(x)
    expected = np.array([1/3, 1/3, 1/3])
    assert np.allclose(result, expected), f"Test 3 failed: expected {expected}, got {result}"
    
    # Test 4: Numerical stability (large numbers)
    x = np.array([1000.0, 1001.0, 1002.0])
    result = softmax(x)
    assert not np.isnan(result).any(), "Test 4 failed: got NaN (numerical instability)"
    assert abs(result.sum() - 1.0) < 1e-6, f"Test 4 failed: sum is {result.sum()}"
    
    # Test 5: Single element
    x = np.array([5.0])
    result = softmax(x)
    assert abs(result[0] - 1.0) < 1e-6, f"Test 5 failed: single element should be 1.0"
    
    print("All tests passed!")
