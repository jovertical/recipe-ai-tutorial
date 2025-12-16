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

from collections.abc import Iterator

def batch_generator(items: list, batch_size: int) -> Iterator[list]:
    # Your solution here (use yield)

    for start in range(0, len(items), batch_size):
        # Determine the end of a "batch" based on the size
        #   1. Calculate based on starting index (e.g. 0) + total batch (e.g. 2) - 1 = 1 (0 + 2 - 1)
        #   2. Ensure that last batch is minimized to maximum ending batch index:
        #      Given that items is [1, 2, 3] and batch_size is 2. the last batch index would end in `2`
        end = min(start + batch_size - 1, len(items) - 1)

        # Return the batched items...
        # We've used a trick in python called "slicing":
        yield items[start:end + 1]

        # Basically this, in long form:
        # batched = []
        # for k in range(start, end + 1):
        #     batched.append(items[k])
        # yield

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
