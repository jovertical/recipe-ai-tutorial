# Problem 1: Two Sum
#
# Given a list of integers and a target, return the indices of two numbers
# that add up to the target.
#
# Example:
#   Input: nums = [2, 7, 11, 15], target = 9
#   Output: [0, 1]  # because nums[0] + nums[1] = 2 + 7 = 9
#
# Constraints:
#   - Each input has exactly one solution
#   - You may not use the same element twice
#   - Aim for O(n) time complexity
#
# ML Relevance: Hash maps are used everywhere - embedding lookups,
# vocabulary mappings, caching computed values.


def two_sum(nums: list[int], target: int) -> list[int]:
    num_map = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i

    return []

# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Basic case
    assert two_sum([2, 7, 11, 15], 9) == [0, 1], "Test 1 failed"

    # Test 2: Target at end
    assert two_sum([3, 2, 4], 6) == [1, 2], "Test 2 failed"

    # Test 3: Negative numbers
    assert two_sum([-1, -2, -3, -4, -5], -8) == [2, 4], "Test 3 failed"

    # Test 4: Larger list
    assert two_sum([1, 5, 8, 3, 9, 2], 11) == [2, 3], "Test 4 failed"

    print("All tests passed!")
