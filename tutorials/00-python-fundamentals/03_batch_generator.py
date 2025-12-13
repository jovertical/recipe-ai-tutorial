# Problem 3: Batch Generator
#
# Create a generator that yields batches of items from a list.
# This is exactly how data loaders work in ML training.
#
# Example:
#   Input: items = [1, 2, 3, 4, 5, 6, 7], batch_size = 3
#   Output (yielded): [1, 2, 3], [4, 5, 6], [7]
#
# Constraints:
#   - Must be a generator (use yield)
#   - Last batch can be smaller than batch_size
#   - batch_size is always > 0
#
# ML Relevance: Training loops process data in batches. Understanding
# generators is key for memory-efficient data loading.


def batch_generator(items: list, batch_size: int):
    # Your solution here (use yield)
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Even split
    result = list(batch_generator([1, 2, 3, 4, 5, 6], 2))
    assert result == [[1, 2], [3, 4], [5, 6]], f"Test 1 failed: {result}"
    
    # Test 2: Uneven split
    result = list(batch_generator([1, 2, 3, 4, 5, 6, 7], 3))
    assert result == [[1, 2, 3], [4, 5, 6], [7]], f"Test 2 failed: {result}"
    
    # Test 3: Batch size larger than list
    result = list(batch_generator([1, 2], 5))
    assert result == [[1, 2]], f"Test 3 failed: {result}"
    
    # Test 4: Batch size of 1
    result = list(batch_generator([1, 2, 3], 1))
    assert result == [[1], [2], [3]], f"Test 4 failed: {result}"
    
    # Test 5: Verify it is a generator
    gen = batch_generator([1, 2, 3], 2)
    assert hasattr(gen, "__next__"), "Must be a generator (use yield)"
    
    print("All tests passed!")
