# Problem 5: Simple Tokenizer
#
# Build a basic tokenizer that converts text into tokens (integers).
# This is the first step in any NLP pipeline.
#
# The tokenizer should:
# 1. Lowercase the text
# 2. Split on whitespace and punctuation
# 3. Build a vocabulary mapping words to unique IDs
# 4. Convert text to list of token IDs
#
# Example:
#   tokenizer = SimpleTokenizer()
#   tokenizer.fit(["Hello world", "Hello there"])
#   # vocab: {"hello": 0, "world": 1, "there": 2}
#
#   tokenizer.encode("Hello world")  # returns [0, 1]
#   tokenizer.decode([0, 1])          # returns "hello world"
#
# Constraints:
#   - Handle unknown words by returning special token <UNK> with ID 0
#   - Punctuation should be separate tokens
#   - Hint: use re.findall(r"[a-z]+|[.,!?;]", text.lower())
#
# ML Relevance: This is literally what tokenizers do. Understanding this
# helps you debug tokenization issues in real models.

import re


class SimpleTokenizer:
    vocab: dict[str, int]
    reverse_vocab: dict[int, str]

    def __init__(self):
        # Reserve ID 0 for <UNK>
        self.vocab = { '<UNK>': 0 }
        self.reverse_vocab = { 0: '<UNK>' }

        pass

    def fit(self, texts: list[str]) -> None:
        """Build vocabulary from list of texts."""

        # We expect the texts to be sentences so we loop through them
        for text in texts:
            # We then extract the words from the sentence (text in this case)
            for word in self.__extract_words(text):
                # We check if the "word" is not within our vocabulary
                if word not in self.vocab:
                    # This is a unique ID O(1)
                    new_id = self.vocab_size

                    self.vocab[word] = new_id
                    self.reverse_vocab[new_id] = word

    def encode(self, text: str) -> list[int]:
        """Convert text to list of token IDs."""

        # 1. Extract the words from the given text
        # 2. Find the token ID from `vocab` mapping
        # 3. Defaults to `{ <UNK>: 0 }` if the token ID is not found
        return [self.vocab.get(word, 0) for word in self.__extract_words(text)]

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""

        # 1. Extract the word from our reverse_vocab using the given token IDs
        # 2. Build the text string by concatenating with a space (" ") between
        return " ".join(self.reverse_vocab[token] for token in token_ids)

    def __extract_words(self, text: str) -> list[str]:
        return re.findall(r"[a-z]+|[.,!?;]", text.lower())

    @property
    def vocab_size(self) -> int:
        """Return size of vocabulary including <UNK>."""
        return len(self.vocab.values())

# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Basic fit and encode
    tokenizer = SimpleTokenizer()
    tokenizer.fit(["hello world", "hello there"])
    encoded = tokenizer.encode("hello world")
    assert len(encoded) == 2, f"Test 1a failed: expected 2 tokens, got {len(encoded)}"
    assert encoded[0] == encoded[0], "Test 1b failed: same word should have same ID"

    # Test 2: Decode
    decoded = tokenizer.decode(encoded)
    assert decoded == "hello world", f"Test 2 failed: got {decoded}"

    # Test 3: Unknown words get <UNK> token (ID 0)
    encoded = tokenizer.encode("hello universe")
    assert 0 in encoded, "Test 3 failed: unknown word should map to 0"

    # Test 4: Punctuation handling
    tokenizer = SimpleTokenizer()
    tokenizer.fit(["hello, world!"])
    encoded = tokenizer.encode("hello, world!")
    assert len(encoded) == 4, f"Test 4 failed: expected 4 tokens (hello , world !), got {len(encoded)}"

    # Test 5: Vocab size
    tokenizer = SimpleTokenizer()
    tokenizer.fit(["a b c"])
    assert tokenizer.vocab_size == 4, f"Test 5 failed: expected 4 (UNK + 3 words), got {tokenizer.vocab_size}"

    print("All tests passed!")
