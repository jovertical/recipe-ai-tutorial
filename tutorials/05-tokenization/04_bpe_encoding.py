# Problem 4: BPE Encoding and Decoding
#
# Now that you can train BPE, implement encoding and decoding.
# Use the learned merge rules to tokenize new text.
#
# Encoding with BPE:
# 1. Start with characters (like training)
# 2. Apply merge rules in the order they were learned
# 3. Continue until no more merges can be applied
#
# Example:
#   merges = [("l", "o"), ("lo", "w"), ("low", "</w>")]
#   encode("low") -> ["low</w>"]  (if we apply all merges)
#   encode("lower") -> ["low", "e", "r", "</w>"]  (partial match)
#
# Constraints:
#   - Apply merges in the exact order they were learned
#   - Handle words not seen during training
#   - Build a complete BPE tokenizer class
#
# ML Relevance: This is how models like GPT tokenize your input text.
# Understanding this helps debug why models sometimes make strange token splits.


class BPETokenizer:
    def __init__(self):
        self.merges: list[tuple[str, str]] = []  # Ordered list of merge rules
        self.vocab: dict[str, int] = {"<UNK>": 0, "<PAD>": 1}
        self.id_to_token: dict[int, str] = {0: "<UNK>", 1: "<PAD>"}

    def train(self, words: list[str], num_merges: int) -> None:
        """
        Train BPE on a list of words.
        Should populate self.merges and build vocabulary.
        """
        # Your solution here
        # Hint: Reuse your train_bpe function from problem 3
        # Then build vocab from final tokens
        pass

    def _tokenize_word(self, word: str) -> list[str]:
        """
        Convert a word to initial character tokens with </w>.
        """
        return list(word) + ["</w>"]

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        """
        Apply all learned merges to a list of tokens.
        Apply merges in order until no more can be applied.

        Example:
            merges = [("l", "o"), ("lo", "w")]
            tokens = ["l", "o", "w", "</w>"]
            -> ["lo", "w", "</w>"]  (after first merge)
            -> ["low", "</w>"]  (after second merge)
        """
        # Your solution here
        pass

    def encode_word(self, word: str) -> list[str]:
        """
        Encode a single word into BPE tokens.

        Example:
            word = "lower"
            -> ["low", "er</w>"] or similar depending on training
        """
        # Your solution here
        pass

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.

        Steps:
        1. Split text into words (on whitespace)
        2. Encode each word with BPE
        3. Convert tokens to IDs (unknown tokens -> <UNK>)
        """
        # Your solution here
        pass

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back to text.

        Steps:
        1. Convert IDs to tokens
        2. Join tokens and remove </w> markers
        3. Handle word boundaries correctly
        """
        # Your solution here
        pass

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def visualize_tokenization(tokenizer: BPETokenizer, text: str) -> str:
    """
    Show how text is tokenized with visible boundaries.

    Example:
        "hello world" -> "|hel|lo</w>| |wor|ld</w>|"
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Train tokenizer
    tokenizer = BPETokenizer()
    words = ["low", "low", "low", "lower", "newest", "widest"]
    tokenizer.train(words, num_merges=10)

    assert len(tokenizer.merges) == 10, f"Test 1a failed: expected 10 merges, got {len(tokenizer.merges)}"
    assert tokenizer.vocab_size > 2, "Test 1b failed: vocab should have more than special tokens"

    # Test 2: Apply merges
    tokenizer2 = BPETokenizer()
    tokenizer2.merges = [("l", "o"), ("lo", "w")]
    tokens = tokenizer2._apply_merges(["l", "o", "w", "</w>"])
    assert tokens == ["low", "</w>"], f"Test 2 failed: got {tokens}"

    # Test 3: Encode word
    tokenizer3 = BPETokenizer()
    tokenizer3.merges = [("l", "o"), ("lo", "w")]
    encoded = tokenizer3.encode_word("low")
    assert "low" in encoded or ("lo" in encoded and "w" in encoded), f"Test 3 failed: got {encoded}"

    # Test 4: Partial merge (word not fully merged)
    tokenizer4 = BPETokenizer()
    tokenizer4.merges = [("l", "o")]
    encoded = tokenizer4.encode_word("low")
    assert "lo" in encoded, f"Test 4 failed: 'lo' should be merged, got {encoded}"

    # Test 5: Full encode/decode roundtrip
    tokenizer = BPETokenizer()
    training_words = ["baking", "baked", "baker", "bake"] * 10  # Repeat for frequency
    tokenizer.train(training_words, num_merges=15)

    # Encode and decode a training word
    text = "baking"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    # Remove potential extra spaces from decode
    decoded = decoded.strip()
    assert decoded == text, f"Test 5 failed: expected '{text}', got '{decoded}'"

    # Test 6: Handle unknown tokens
    tokenizer = BPETokenizer()
    tokenizer.train(["hello", "world"], num_merges=5)
    encoded = tokenizer.encode("xyz")  # Unknown word
    assert 0 in encoded or any(
        tokenizer.id_to_token.get(i, "").startswith("<") for i in encoded
    ), "Test 6 failed: unknown tokens should be handled"

    # Test 7: Vocabulary includes special tokens
    tokenizer = BPETokenizer()
    tokenizer.train(["test"], num_merges=3)
    assert "<UNK>" in tokenizer.vocab, "Test 7a failed: <UNK> should be in vocab"
    assert "<PAD>" in tokenizer.vocab, "Test 7b failed: <PAD> should be in vocab"

    # Test 8: Visualize tokenization
    tokenizer = BPETokenizer()
    tokenizer.merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
    viz = visualize_tokenization(tokenizer, "hello")
    assert "|" in viz, f"Test 8 failed: visualization should show boundaries, got {viz}"

    # Test 9: Recipe-like text
    tokenizer = BPETokenizer()
    recipe_words = [
        "preheat", "preheating", "heat", "heating",
        "bake", "baking", "baked",
        "mix", "mixing", "mixed"
    ] * 5
    tokenizer.train(recipe_words, num_merges=20)

    # Common prefixes should be learned
    encoded = tokenizer.encode("baking")
    assert len(encoded) < 7, f"Test 9 failed: 'baking' should be fewer than 7 tokens, got {len(encoded)}"

    print("All tests passed!")
