# Problem 2: Matrix Transpose
#
# Given a 2D matrix (list of lists), return its transpose.
# The transpose flips a matrix over its diagonal - rows become columns.
#
# Example:
#   Input:  [[1, 2, 3],
#            [4, 5, 6]]
#   Output: [[1, 4],
#            [2, 5],
#            [3, 6]]
#
# Constraints:
#   - Matrix can be any MxN size (not necessarily square)
#   - Try using list comprehension
#
# ML Relevance: Tensor operations - transposing weight matrices,
# reshaping data between batch-first and sequence-first formats.


def transpose(matrix: list[list[int]]) -> list[list[int]]:
    transposed = []

    # Generate a list of column indices
    #   1. Using the first row of the matrix, get the number of columns -> e.g. [1, 2, 3]
    #   2. Use range() to create a list of column indices -> e.g. [0, 1, 2]
    for i in range(len(matrix[0])):
        # Generate each transposed row
        #  1. For each column index, iterate over all rows -> e.g. for column index 0
        #  2. Collect the elements at that column index from each row -> e.g. [1, 4]
        for j in range(len(matrix)):
            transposed.append(matrix[j][i])

    # Reshape the flat list into a list of lists (a.k.a. chunking) using list comprehension
    transposed = [transposed[k:k + len(matrix)] for k in range(0, len(transposed), len(matrix))]

    # Alternative way without list comprehension
    # result = []
    # for k in range(0, len(transposed), len(matrix)):
    #     result.append(transposed[k:k + len(matrix)])
    # return result

    return transposed

# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: 2x3 matrix
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]], "Test 1 failed"

    # Test 2: 3x3 square matrix
    assert transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[1, 4, 7], [2, 5, 8], [3, 6, 9]], "Test 2 failed"

    # Test 3: 1x4 matrix (row vector)
    assert transpose([[1, 2, 3, 4]]) == [[1], [2], [3], [4]], "Test 3 failed"

    # Test 4: 4x1 matrix (column vector)
    assert transpose([[1], [2], [3], [4]]) == [[1, 2, 3, 4]], "Test 4 failed"

    print("All tests passed!")
