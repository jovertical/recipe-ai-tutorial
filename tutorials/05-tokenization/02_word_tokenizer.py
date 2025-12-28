# Problem 2: Word Tokenizer
#
# Build a word-level tokenizer with vocabulary limits.
# This is the opposite extreme from character tokenization.
#
# The tokenizer should:
# 1. Split text into words (on whitespace and punctuation)
# 2. Build a vocabulary with a maximum size limit
# 3. Keep only the most frequent words
# 4. Handle out-of-vocabulary (OOV) words with <UNK>
#
# Example:
#   tokenizer = WordTokenizer(max_vocab_size=100)
#   tokenizer.build_vocab(["I love cooking", "I love baking"])
#   # vocab: {'<UNK>': 0, '<PAD>': 1, 'I': 2, 'love': 3, 'cooking': 4, 'baking': 5}
#
#   tokenizer.encode("I love eating")  # returns [2, 3, 0]  (eating -> <UNK>)
#
# Constraints:
#   - Use regex to split: words and punctuation as separate tokens
#   - max_vocab_size includes special tokens (<UNK>, <PAD>)
#   - When vocab limit is reached, keep most frequent words
#   - Tie-breaking: keep words that appeared first
#
# ML Relevance: Word tokenization has a huge vocabulary (100k+ words) but short
# sequences. Rare words become <UNK>, losing information. This trade-off is why
# subword methods like BPE were invented.

import re
from collections import Counter


class WordTokenizer:
    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.word_to_id: dict[str, int] = {"<UNK>": 0, "<PAD>": 1}
        self.id_to_word: dict[int, str] = {0: "<UNK>", 1: "<PAD>"}

    def _tokenize(self, text: str) -> list[str]:
        """
        Split text into words and punctuation.
        Use regex to find: sequences of letters/numbers OR single punctuation marks.

        Example: "Hello, world!" -> ["Hello", ",", "world", "!"]
        """
        # Your solution here
        # Hint: re.findall(r"[A-Za-z0-9]+|[.,!?;:'\"-]", text)
        pass

    def build_vocab(self, texts: list[str]) -> None:
        """
        Build vocabulary from texts, respecting max_vocab_size.

        Steps:
        1. Tokenize all texts
        2. Count word frequencies
        3. Keep top (max_vocab_size - 2) words (reserve 2 for special tokens)
        4. Assign IDs starting from 2
        """
        # Your solution here
        pass

    def encode(self, text: str) -> list[int]:
        """
        Convert text to token IDs.
        Unknown words map to <UNK> (ID 0).
        """
        # Your solution here
        pass

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back to text.
        Join with spaces. Skip <PAD> tokens.
        """
        # Your solution here
        pass

    @property
    def vocab_size(self) -> int:
        """Return actual vocabulary size."""
        return len(self.word_to_id)

    def get_oov_rate(self, text: str) -> float:
        """
        Calculate the out-of-vocabulary rate for a text.
        Returns the fraction of tokens that map to <UNK>.
        """
        # Your solution here
        pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Basic tokenization
    tokenizer = WordTokenizer()
    tokens = tokenizer._tokenize("Hello, world!")
    assert tokens == ["Hello", ",", "world", "!"], f"Test 1 failed: got {tokens}"

    # Test 2: Build vocabulary
    tokenizer = WordTokenizer(max_vocab_size=10)
    tokenizer.build_vocab([
        "the cat sat on the mat",
        "the dog sat on the rug"
    ])
    assert tokenizer.vocab_size <= 10, f"Test 2a failed: vocab size {tokenizer.vocab_size} > 10"
    assert "the" in tokenizer.word_to_id, "Test 2b failed: 'the' should be in vocab (most frequent)"

    # Test 3: Vocabulary limit enforced
    tokenizer = WordTokenizer(max_vocab_size=5)  # Only room for 3 words + 2 special
    tokenizer.build_vocab(["a b c d e f g h"])
    assert tokenizer.vocab_size == 5, f"Test 3 failed: expected vocab size 5, got {tokenizer.vocab_size}"

    # Test 4: Frequency-based selection
    tokenizer = WordTokenizer(max_vocab_size=4)  # Room for 2 words
    tokenizer.build_vocab(["rare common common common rare common"])
    assert "common" in tokenizer.word_to_id, "Test 4 failed: 'common' should be kept (more frequent)"

    # Test 5: Encode and decode
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(["I love recipes"])
    encoded = tokenizer.encode("I love recipes")
    decoded = tokenizer.decode(encoded)
    assert decoded == "I love recipes", f"Test 5 failed: got '{decoded}'"

    # Test 6: OOV handling
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(["hello world"])
    encoded = tokenizer.encode("hello universe")
    assert 0 in encoded, "Test 6a failed: 'universe' should map to <UNK>"
    oov_rate = tokenizer.get_oov_rate("hello universe")
    assert oov_rate == 0.5, f"Test 6b failed: expected OOV rate 0.5, got {oov_rate}"

    # Test 7: Skip PAD in decode
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(["hello world"])
    hello_id = tokenizer.word_to_id["hello"]
    decoded = tokenizer.decode([hello_id, 1, 1, 1])  # 1 is <PAD>
    assert decoded == "hello", f"Test 7 failed: expected 'hello', got '{decoded}'"

    # Test 8: Recipe text
    tokenizer = WordTokenizer(max_vocab_size=50)
    recipes = [
        "Preheat oven to 350 degrees.",
        "Mix flour, sugar, and salt.",
        "Add eggs and butter.",
        "Bake for 30 minutes."
    ]
    tokenizer.build_vocab(recipes)
    encoded = tokenizer.encode("Mix flour and eggs.")
    assert len(encoded) == 5, f"Test 8 failed: expected 5 tokens, got {len(encoded)}"

    print("All tests passed!")
