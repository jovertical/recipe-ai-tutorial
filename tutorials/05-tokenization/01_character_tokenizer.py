# Problem 1: Character Tokenizer
#
# Build a character-level tokenizer - the simplest form of tokenization.
# Each character becomes its own token.
#
# The tokenizer should:
# 1. Build a vocabulary from a corpus of text
# 2. Map each unique character to a unique integer ID
# 3. Encode text into a list of token IDs
# 4. Decode token IDs back into text
#
# Example:
#   tokenizer = CharTokenizer()
#   tokenizer.build_vocab(["hello", "world"])
#   # vocab might be: {'h': 0, 'e': 1, 'l': 2, 'o': 3, 'w': 4, 'r': 5, 'd': 6, ' ': 7}
#
#   tokenizer.encode("hello")  # returns [0, 1, 2, 2, 3]
#   tokenizer.decode([0, 1, 2, 2, 3])  # returns "hello"
#
# Constraints:
#   - Handle unknown characters with a special <UNK> token (ID 0)
#   - Include a <PAD> token (ID 1) for padding sequences
#   - Vocabulary should be built in order of first appearance
#
# ML Relevance: Character tokenization is the simplest approach. It has a tiny
# vocabulary (just ~100 characters) but creates very long sequences. Understanding
# this trade-off helps you appreciate why subword tokenization exists.


class CharTokenizer:
    def __init__(self):
        # Initialize with special tokens
        # <UNK> = unknown character (ID 0)
        # <PAD> = padding token (ID 1)
        self.char_to_id: dict[str, int] = {"<UNK>": 0, "<PAD>": 1}
        self.id_to_char: dict[int, str] = {0: "<UNK>", 1: "<PAD>"}

    def build_vocab(self, texts: list[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Each unique character gets a unique ID.
        Characters are added in order of first appearance.
        """
        # Your solution here
        pass

    def encode(self, text: str) -> list[int]:
        """
        Convert text to a list of token IDs.
        Unknown characters should map to <UNK> (ID 0).
        """
        # Your solution here
        pass

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back to text.
        Skip <PAD> tokens in output.
        <UNK> tokens should be rendered as '?'.
        """
        # Your solution here
        pass

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.char_to_id)


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Build vocabulary
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["hello", "world"])

    assert tokenizer.vocab_size > 2, "Test 1a failed: vocab should have more than special tokens"
    assert "h" in tokenizer.char_to_id, "Test 1b failed: 'h' should be in vocab"
    assert "z" not in tokenizer.char_to_id, "Test 1c failed: 'z' should not be in vocab"

    # Test 2: Encode basic text
    encoded = tokenizer.encode("hello")
    assert len(encoded) == 5, f"Test 2a failed: expected 5 tokens, got {len(encoded)}"
    assert encoded[2] == encoded[3], "Test 2b failed: both 'l' chars should have same ID"

    # Test 3: Decode back to text
    decoded = tokenizer.decode(encoded)
    assert decoded == "hello", f"Test 3 failed: expected 'hello', got '{decoded}'"

    # Test 4: Handle unknown characters
    encoded_with_unk = tokenizer.encode("hello!")
    assert 0 in encoded_with_unk, "Test 4a failed: '!' should map to <UNK> (0)"
    decoded_with_unk = tokenizer.decode(encoded_with_unk)
    assert decoded_with_unk == "hello?", f"Test 4b failed: expected 'hello?', got '{decoded_with_unk}'"

    # Test 5: Skip PAD tokens in decode
    padded = [tokenizer.char_to_id["h"], tokenizer.char_to_id["i"], 1, 1, 1]
    decoded_padded = tokenizer.decode(padded)
    assert decoded_padded == "hi", f"Test 5 failed: expected 'hi', got '{decoded_padded}'"

    # Test 6: Encode-decode roundtrip
    tokenizer2 = CharTokenizer()
    tokenizer2.build_vocab(["The quick brown fox jumps over the lazy dog."])
    original = "The fox"
    roundtrip = tokenizer2.decode(tokenizer2.encode(original))
    assert roundtrip == original, f"Test 6 failed: roundtrip failed, got '{roundtrip}'"

    # Test 7: Empty text
    assert tokenizer.encode("") == [], "Test 7a failed: empty text should give empty list"
    assert tokenizer.decode([]) == "", "Test 7b failed: empty list should give empty string"

    print("All tests passed!")
