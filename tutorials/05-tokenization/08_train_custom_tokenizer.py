# Problem 8: Train a Custom Tokenizer
#
# Train a BPE tokenizer specifically for recipe text using Hugging Face's
# tokenizers library. This is the production way to create custom tokenizers.
#
# Why train a custom tokenizer?
# 1. Better compression for domain-specific text
# 2. Fewer tokens for common terms (ingredients, cooking verbs)
# 3. More meaningful subword splits
#
# Example comparison:
#   Generic tokenizer: "sauté" -> ["s", "aut", "é"] (3 tokens)
#   Recipe tokenizer:  "sauté" -> ["sauté"] (1 token)
#
# Prerequisites:
#   pip install tokenizers
#
# ML Relevance: Training custom tokenizers is common in production.
# Companies like OpenAI, Meta, and Google all train tokenizers specifically
# for their data distributions. This skill is directly applicable.

from pathlib import Path
import json

# Use the tokenizers library (faster than transformers for training)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def prepare_training_corpus(recipes: list[dict], output_path: str) -> str:
    """
    Prepare recipe data as a text file for tokenizer training.

    Args:
        recipes: List of recipe dicts with 'title', 'ingredients', 'instructions'
        output_path: Where to save the training corpus

    Returns:
        Path to the saved corpus file

    Each recipe should be formatted as plain text, one per line or
    with clear separators. Include:
    - Recipe titles
    - Ingredient lists
    - Instruction steps
    """
    # Your solution here
    pass


def train_bpe_tokenizer(
    corpus_path: str,
    vocab_size: int = 8000,
    min_frequency: int = 2,
    special_tokens: list[str] = None
) -> Tokenizer:
    """
    Train a BPE tokenizer on the corpus.

    Args:
        corpus_path: Path to training text file
        vocab_size: Target vocabulary size
        min_frequency: Minimum token frequency to include
        special_tokens: List of special tokens to add

    Returns:
        Trained Tokenizer object
    """
    # Your solution here
    # 1. Create BPE tokenizer
    # 2. Create trainer with vocab_size and special_tokens
    # 3. Set pre-tokenizer (Whitespace is simple and works well)
    # 4. Train on corpus
    pass


def add_post_processor(tokenizer: Tokenizer) -> Tokenizer:
    """
    Add post-processing to handle special tokens properly.

    This ensures:
    - BOS token is added at the start
    - EOS token is added at the end

    Returns the modified tokenizer.
    """
    # Your solution here
    # Use TemplateProcessing
    pass


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    """
    Save the trained tokenizer to disk.
    """
    # Your solution here
    pass


def load_tokenizer(path: str) -> Tokenizer:
    """
    Load a saved tokenizer from disk.
    """
    # Your solution here
    pass


def evaluate_tokenizer(
    tokenizer: Tokenizer,
    test_texts: list[str],
    compare_tokenizer: Tokenizer = None
) -> dict:
    """
    Evaluate tokenizer quality on test texts.

    Returns:
    - 'avg_tokens_per_text': average sequence length
    - 'avg_fertility': tokens per word
    - 'vocab_size': vocabulary size
    - 'unknown_rate': fraction of <UNK> tokens
    - 'compression_ratio': chars per token

    If compare_tokenizer is provided, also include:
    - 'tokens_saved': % fewer tokens vs comparison
    """
    # Your solution here
    pass


def analyze_vocabulary(tokenizer: Tokenizer, top_n: int = 50) -> dict:
    """
    Analyze what the tokenizer learned.

    Returns:
    - 'vocab_size': total vocabulary size
    - 'top_tokens': most common tokens (by ID, lower = more common in training)
    - 'recipe_terms': tokens that look like cooking terms
    - 'ingredient_tokens': tokens that look like ingredients
    - 'subword_examples': examples of subword splits
    """
    # Your solution here
    pass


def convert_to_huggingface(tokenizer: Tokenizer, save_path: str) -> None:
    """
    Convert the tokenizers library tokenizer to Hugging Face format.

    This allows using it with transformers models.

    Saves files that can be loaded with:
    AutoTokenizer.from_pretrained(save_path)
    """
    # Your solution here
    pass


# Sample recipe data for testing
SAMPLE_RECIPES = [
    {
        "title": "Classic Chocolate Chip Cookies",
        "ingredients": ["2 cups flour", "1 cup butter", "1 cup sugar", "2 eggs", "1 cup chocolate chips"],
        "instructions": ["Preheat oven to 375F", "Cream butter and sugar", "Add eggs", "Mix in flour", "Fold in chips", "Bake 10 minutes"]
    },
    {
        "title": "Simple Tomato Pasta",
        "ingredients": ["1 lb pasta", "2 cups tomato sauce", "3 cloves garlic", "olive oil", "basil"],
        "instructions": ["Cook pasta", "Sauté garlic in oil", "Add tomato sauce", "Simmer 10 minutes", "Toss with pasta", "Garnish with basil"]
    },
    {
        "title": "Fluffy Pancakes",
        "ingredients": ["2 cups flour", "2 tbsp sugar", "1 tbsp baking powder", "2 eggs", "1.5 cups milk", "butter"],
        "instructions": ["Mix dry ingredients", "Whisk wet ingredients", "Combine gently", "Heat griddle", "Pour batter", "Flip when bubbly", "Serve with syrup"]
    },
    {
        "title": "Garlic Butter Shrimp",
        "ingredients": ["1 lb shrimp", "4 tbsp butter", "6 cloves garlic", "lemon juice", "parsley"],
        "instructions": ["Melt butter", "Sauté garlic", "Add shrimp", "Cook 2-3 minutes per side", "Add lemon juice", "Garnish with parsley"]
    },
    {
        "title": "Caesar Salad",
        "ingredients": ["romaine lettuce", "caesar dressing", "croutons", "parmesan cheese"],
        "instructions": ["Chop lettuce", "Add dressing", "Toss well", "Top with croutons", "Shave parmesan on top"]
    }
]


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    import tempfile
    import os

    # Create temp directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        tokenizer_path = os.path.join(tmpdir, "tokenizer.json")

        # Test 1: Prepare corpus
        saved_path = prepare_training_corpus(SAMPLE_RECIPES, corpus_path)
        assert os.path.exists(saved_path), "Test 1a failed: corpus file not created"
        with open(saved_path) as f:
            content = f.read()
        assert len(content) > 100, "Test 1b failed: corpus too short"
        assert "chocolate" in content.lower(), "Test 1c failed: corpus should contain recipe content"

        # Test 2: Train tokenizer
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        tokenizer = train_bpe_tokenizer(
            corpus_path,
            vocab_size=500,  # Small for testing
            min_frequency=1,
            special_tokens=special_tokens
        )
        assert tokenizer is not None, "Test 2a failed: tokenizer is None"
        assert tokenizer.get_vocab_size() <= 500, f"Test 2b failed: vocab size {tokenizer.get_vocab_size()} > 500"

        # Test 3: Tokenizer can encode/decode
        test_text = "Mix flour and sugar"
        encoded = tokenizer.encode(test_text)
        assert len(encoded.ids) > 0, "Test 3a failed: encoding produced no tokens"
        decoded = tokenizer.decode(encoded.ids)
        assert "flour" in decoded.lower(), f"Test 3b failed: decode failed, got '{decoded}'"

        # Test 4: Save and load
        save_tokenizer(tokenizer, tokenizer_path)
        assert os.path.exists(tokenizer_path), "Test 4a failed: tokenizer not saved"

        loaded = load_tokenizer(tokenizer_path)
        assert loaded is not None, "Test 4b failed: could not load tokenizer"

        # Verify loaded tokenizer works
        encoded2 = loaded.encode(test_text)
        assert encoded.ids == encoded2.ids, "Test 4c failed: loaded tokenizer gives different results"

        # Test 5: Evaluate tokenizer
        test_texts = [
            "Preheat the oven to 350 degrees",
            "Mix the flour and sugar together",
            "Bake for 20 minutes until golden"
        ]
        evaluation = evaluate_tokenizer(tokenizer, test_texts)
        assert "avg_tokens_per_text" in evaluation, "Test 5a failed: missing avg_tokens_per_text"
        assert "compression_ratio" in evaluation, "Test 5b failed: missing compression_ratio"
        assert evaluation["avg_tokens_per_text"] > 0, "Test 5c failed: invalid avg_tokens_per_text"

        # Test 6: Analyze vocabulary
        analysis = analyze_vocabulary(tokenizer, top_n=20)
        assert "vocab_size" in analysis, "Test 6a failed: missing vocab_size"
        assert "top_tokens" in analysis, "Test 6b failed: missing top_tokens"

        # Test 7: Special tokens are in vocabulary
        vocab = tokenizer.get_vocab()
        for special in special_tokens:
            assert special in vocab, f"Test 7 failed: {special} not in vocab"

        # Test 8: Recipe-specific terms should be well-tokenized
        # Common cooking words should ideally be single tokens after training
        cooking_words = ["flour", "sugar", "butter", "bake"]
        good_tokenization_count = 0
        for word in cooking_words:
            encoded = tokenizer.encode(word)
            if len(encoded.ids) <= 2:  # Single token or 2 subwords is good
                good_tokenization_count += 1

        # At least half should be well-tokenized
        assert good_tokenization_count >= 2, \
            f"Test 8 failed: only {good_tokenization_count}/4 cooking words well-tokenized"

        # Test 9: Larger vocabulary training
        tokenizer_large = train_bpe_tokenizer(
            corpus_path,
            vocab_size=1000,
            min_frequency=1,
            special_tokens=special_tokens
        )
        assert tokenizer_large.get_vocab_size() <= 1000, "Test 9 failed: large tokenizer vocab too big"

        # Test 10: Unknown handling
        unknown_text = "xyzzy frobozz"  # Made-up words
        encoded_unk = tokenizer.encode(unknown_text)
        # Should still produce some tokens (character fallback or UNK)
        assert len(encoded_unk.ids) > 0, "Test 10 failed: should handle unknown words"

    print("All tests passed!")
    print("\nYour custom tokenizer is ready!")
    print("Next steps:")
    print("1. Train on your full recipe dataset")
    print("2. Experiment with different vocab sizes (4k-32k)")
    print("3. Use convert_to_huggingface() for model training")
