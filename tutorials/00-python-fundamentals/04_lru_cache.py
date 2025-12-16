# Problem 4: LRU Cache
#
# Implement a Least Recently Used (LRU) cache with a max capacity.
# When the cache is full and a new item is added, remove the least
# recently used item.
#
# Example:
#   cache = LRUCache(2)  # capacity of 2
#   cache.put("a", 1)    # cache: {a: 1}
#   cache.put("b", 2)    # cache: {a: 1, b: 2}
#   cache.get("a")       # returns 1, cache: {b: 2, a: 1} (a is now most recent)
#   cache.put("c", 3)    # cache: {a: 1, c: 3} (b was evicted, least recent)
#   cache.get("b")       # returns None (was evicted)
#
# Constraints:
#   - get(key) returns value or None if not found
#   - put(key, value) adds or updates the key
#   - Both operations should be O(1)
#   - Hint: Python has OrderedDict
#
# ML Relevance: Caching embeddings, model outputs, tokenization results.
# Understanding cache eviction helps with memory management.

from collections import OrderedDict
from typing import Any

class LRUCache:
    store: OrderedDict[str, Any]
    capacity: int

    def __init__(self, capacity: int):
        self.capacity = capacity

        # Initialize the store using ordered dict so we can conditionally remove
        # least accessed items from the store
        self.store = OrderedDict()

    def get(self, key: str) -> Any:
        item = self.store.get(key)

         # Push the item in the end so when max capacity is reached
         # We can remove the first one in the store when putting a new item
        if item is not None:
            self.store.move_to_end(key)

        return item

    def put(self, key: str, value) -> None:
        self.store[key] = value

        # If we reach the capacity, we remove the first item from the store
        # When popping an item, we should use FIFO since the latest is the most
        # recent used cache item
        if len(self.store.items()) > self.capacity:
            self.store.popitem(False)

# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Basic put and get
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.get("a") == 1, "Test 1a failed"
    assert cache.get("b") == 2, "Test 1b failed"

    # Test 2: Eviction
    cache.put("c", 3)  # Should evict "a" since "b" was accessed more recently
    assert cache.get("a") is None, "Test 2a failed - a should be evicted"
    assert cache.get("b") == 2, "Test 2b failed"
    assert cache.get("c") == 3, "Test 2c failed"

    # Test 3: Update existing key
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)  # Update a
    assert cache.get("a") == 10, "Test 3 failed"

    # Test 4: Access refreshes recency
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")      # Access a, making it most recent
    cache.put("c", 3)   # Should evict b, not a
    assert cache.get("a") == 1, "Test 4a failed - a should still exist"
    assert cache.get("b") is None, "Test 4b failed - b should be evicted"

    print("All tests passed!")
