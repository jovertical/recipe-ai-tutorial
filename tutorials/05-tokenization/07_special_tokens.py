# Problem 7: Special Tokens for Recipes
#
# Add domain-specific special tokens to help models understand recipe structure.
# Special tokens mark semantic boundaries that matter for your task.
#
# Recipe-specific tokens we'll add:
# - [TITLE] - marks the recipe title
# - [INGREDIENTS] - marks the start of ingredients
# - [INSTRUCTIONS] - marks the start of instructions
# - [STEP] - marks each instruction step
# - [END] - marks the end of the recipe
#
# Example:
#   "[TITLE] Chocolate Cookies [INGREDIENTS] flour, sugar, chocolate [INSTRUCTIONS] [STEP] Mix dry ingredients [STEP] Bake at 350F [END]"
#
# Why this matters:
# - Models can learn to generate structured output
# - Easier to parse model outputs
# - Better understanding of recipe structure
#
# ML Relevance: Special tokens are used everywhere - [CLS], [SEP] in BERT,
# <|im_start|> in ChatML, etc. Learning to add custom tokens is essential
# for adapting models to new domains.

from transformers import AutoTokenizer


# Define our special tokens
RECIPE_SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "[TITLE]",
        "[INGREDIENTS]",
        "[INSTRUCTIONS]",
        "[STEP]",
        "[END]",
        "[INGREDIENT]",  # For marking individual ingredients
        "[QUANTITY]",    # For marking quantities
        "[UNIT]",        # For marking units
    ]
}


def add_recipe_tokens(tokenizer) -> tuple:
    """
    Add recipe-specific special tokens to a tokenizer.

    Args:
        tokenizer: A Hugging Face tokenizer

    Returns:
        Tuple of (updated_tokenizer, num_added_tokens)

    Note: This modifies the tokenizer in place AND returns it
    """
    # Your solution here
    # Use tokenizer.add_special_tokens()
    pass


def format_recipe_for_training(
    title: str,
    ingredients: list[str],
    instructions: list[str]
) -> str:
    """
    Format a recipe using our special tokens.

    Example output:
    "[TITLE] Chocolate Cookies [INGREDIENTS] [INGREDIENT] 2 cups flour [INGREDIENT] 1 cup sugar [INSTRUCTIONS] [STEP] Preheat oven to 350F [STEP] Mix ingredients [END]"
    """
    # Your solution here
    pass


def parse_formatted_recipe(text: str) -> dict:
    """
    Parse a recipe formatted with special tokens back into structured data.

    Returns:
    - 'title': the recipe title
    - 'ingredients': list of ingredients
    - 'instructions': list of instruction steps

    This is the inverse of format_recipe_for_training.
    """
    # Your solution here
    pass


def resize_model_embeddings(model, tokenizer) -> int:
    """
    Resize a model's embedding layer to accommodate new tokens.

    When you add special tokens to a tokenizer, you must also resize
    the model's embedding matrix to have rows for the new tokens.

    Args:
        model: A Hugging Face model
        tokenizer: The tokenizer with added tokens

    Returns:
        The new embedding size

    Note: New token embeddings are initialized randomly and should be
    fine-tuned on your data.
    """
    # Your solution here
    # Use model.resize_token_embeddings()
    pass


def verify_special_tokens(tokenizer, text_with_tokens: str) -> dict:
    """
    Verify that special tokens are handled correctly.

    Returns:
    - 'tokens': the tokenized output
    - 'special_token_count': number of special tokens found
    - 'special_tokens_preserved': whether each special token is a single token
    - 'issues': list of any problems found
    """
    # Your solution here
    pass


def create_recipe_prompt_template(tokenizer) -> str:
    """
    Create a prompt template for recipe generation using special tokens.

    The template should work for:
    1. Generating a recipe from a title
    2. Completing a partial recipe
    3. Generating instructions from ingredients

    Returns a template string with placeholders.
    """
    # Your solution here
    pass


def get_special_token_ids(tokenizer) -> dict:
    """
    Get the token IDs for all our special tokens.

    Returns dict mapping token string to ID.
    Useful for finding token positions in generated output.
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Load a base tokenizer for testing
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        print("Please run: pip install transformers")
        exit(1)

    original_vocab_size = len(tokenizer)

    # Test 1: Add special tokens
    updated_tokenizer, num_added = add_recipe_tokens(tokenizer)
    assert num_added > 0, f"Test 1a failed: should add tokens, got {num_added}"
    assert len(updated_tokenizer) > original_vocab_size, "Test 1b failed: vocab should grow"
    assert "[TITLE]" in updated_tokenizer.additional_special_tokens, "Test 1c failed: [TITLE] should be added"

    # Test 2: Special tokens are single tokens
    token_ids = updated_tokenizer.encode("[TITLE]", add_special_tokens=False)
    assert len(token_ids) == 1, f"Test 2 failed: [TITLE] should be single token, got {token_ids}"

    # Test 3: Format recipe
    formatted = format_recipe_for_training(
        title="Test Cookies",
        ingredients=["flour", "sugar"],
        instructions=["Mix", "Bake"]
    )
    assert "[TITLE]" in formatted, "Test 3a failed: should contain [TITLE]"
    assert "[INGREDIENTS]" in formatted, "Test 3b failed: should contain [INGREDIENTS]"
    assert "[INSTRUCTIONS]" in formatted, "Test 3c failed: should contain [INSTRUCTIONS]"
    assert "[STEP]" in formatted, "Test 3d failed: should contain [STEP]"
    assert "[END]" in formatted, "Test 3e failed: should contain [END]"

    # Test 4: Parse formatted recipe
    formatted = format_recipe_for_training(
        title="Cookies",
        ingredients=["flour", "sugar"],
        instructions=["Mix dry", "Add wet"]
    )
    parsed = parse_formatted_recipe(formatted)
    assert parsed["title"] == "Cookies", f"Test 4a failed: got title '{parsed['title']}'"
    assert len(parsed["ingredients"]) == 2, f"Test 4b failed: got {len(parsed['ingredients'])} ingredients"
    assert len(parsed["instructions"]) == 2, f"Test 4c failed: got {len(parsed['instructions'])} instructions"

    # Test 5: Verify special tokens
    test_text = "[TITLE] Test [INGREDIENTS] item1 [END]"
    verification = verify_special_tokens(updated_tokenizer, test_text)
    assert "special_token_count" in verification, "Test 5a failed: should count special tokens"
    assert verification["special_token_count"] >= 3, f"Test 5b failed: should find 3+ special tokens"

    # Test 6: Get special token IDs
    token_ids = get_special_token_ids(updated_tokenizer)
    assert "[TITLE]" in token_ids, "Test 6a failed: should have [TITLE] ID"
    assert "[END]" in token_ids, "Test 6b failed: should have [END] ID"
    assert isinstance(token_ids["[TITLE]"], int), "Test 6c failed: ID should be int"

    # Test 7: Tokenize and detokenize preserves special tokens
    formatted = "[TITLE] My Recipe [END]"
    encoded = updated_tokenizer.encode(formatted)
    decoded = updated_tokenizer.decode(encoded)
    assert "[TITLE]" in decoded, f"Test 7a failed: [TITLE] lost in roundtrip, got '{decoded}'"
    assert "[END]" in decoded, f"Test 7b failed: [END] lost in roundtrip, got '{decoded}'"

    # Test 8: Special tokens don't get split
    text = "[TITLE][INGREDIENTS][END]"  # No spaces
    encoded = updated_tokenizer.encode(text, add_special_tokens=False)
    # Should be 3 tokens, not more
    assert len(encoded) == 3, f"Test 8 failed: should be 3 tokens, got {len(encoded)}"

    # Test 9: Prompt template
    template = create_recipe_prompt_template(updated_tokenizer)
    assert isinstance(template, str), "Test 9a failed: should return string"
    assert len(template) > 0, "Test 9b failed: template should not be empty"

    # Test 10: Real recipe formatting
    real_recipe = format_recipe_for_training(
        title="Classic Chocolate Chip Cookies",
        ingredients=[
            "2 cups all-purpose flour",
            "1 cup butter, softened",
            "1 cup chocolate chips"
        ],
        instructions=[
            "Preheat oven to 375°F",
            "Cream butter and sugars",
            "Add flour mixture",
            "Fold in chocolate chips",
            "Bake for 10-12 minutes"
        ]
    )
    assert "Classic Chocolate Chip Cookies" in real_recipe, "Test 10a failed"
    assert real_recipe.count("[STEP]") == 5, f"Test 10b failed: should have 5 steps"

    print("All tests passed!")
