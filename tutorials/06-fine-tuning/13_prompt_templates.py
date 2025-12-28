# Problem 13: Prompt Templates for Training
#
# Format your data correctly for LLM fine-tuning.
# The prompt template significantly affects model behavior!
#
# Common formats:
# 1. Alpaca/Instruction format:
#    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
#
# 2. ChatML format:
#    "<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
#
# 3. Llama-2 Chat format:
#    "[INST] {message} [/INST] {response}"
#
# For recipes, we need templates that:
# - Clearly separate ingredients from instructions
# - Handle variable-length lists
# - Match the format we want at inference time
#
# ML Relevance: Wrong template = wrong model behavior.
# The model learns the pattern, so inference must match training.


def instruction_template(instruction: str, response: str = "") -> str:
    """
    Create Alpaca-style instruction format.

    Args:
        instruction: The task/question
        response: The expected response (empty for inference)

    Returns:
        Formatted string

    Example:
        instruction_template(
            "Write a recipe for cookies",
            "Here's a recipe for chocolate chip cookies..."
        )
        ->
        "### Instruction:
        Write a recipe for cookies

        ### Response:
        Here's a recipe for chocolate chip cookies..."
    """
    # Your solution here
    pass


def instruction_with_input_template(
    instruction: str,
    input_text: str,
    response: str = ""
) -> str:
    """
    Alpaca format with additional input context.

    Useful when you have both an instruction and context.

    Example:
        instruction_with_input_template(
            "Suggest a substitute for the following ingredient",
            "butter",
            "You can substitute butter with..."
        )
    """
    # Your solution here
    pass


def chat_template(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """
    Create ChatML format from messages.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str}
        add_generation_prompt: If True, add prompt for assistant to continue

    Returns:
        Formatted chat string

    Example:
        chat_template([
            {"role": "user", "content": "How do I make pasta?"},
            {"role": "assistant", "content": "Here's how..."}
        ])
        ->
        "<|im_start|>user
        How do I make pasta?<|im_end|>
        <|im_start|>assistant
        Here's how...<|im_end|>"
    """
    # Your solution here
    pass


def llama_chat_template(messages: list[dict]) -> str:
    """
    Create Llama-2 chat format.

    Uses [INST] and [/INST] tags.
    System messages go in <<SYS>> tags.
    """
    # Your solution here
    pass


def recipe_template(
    title: str,
    ingredients: list[str],
    instructions: list[str],
    template_style: str = "instruction"
) -> str:
    """
    Create a formatted recipe for training.

    Args:
        title: Recipe title
        ingredients: List of ingredients
        instructions: List of instruction steps
        template_style: "instruction", "chat", or "structured"

    Returns:
        Formatted training example
    """
    # Your solution here
    pass


def recipe_generation_prompt(
    title: str = None,
    ingredients: list[str] = None,
    cuisine: str = None,
    dietary: str = None,
    template_style: str = "instruction"
) -> str:
    """
    Create a prompt for recipe generation (inference time).

    Supports different generation scenarios:
    - Generate from title only
    - Generate from ingredients
    - Generate with constraints (cuisine, dietary)
    """
    # Your solution here
    pass


def create_training_example(
    recipe: dict,
    task_type: str = "generate"
) -> dict:
    """
    Create a complete training example from a recipe.

    Args:
        recipe: Dict with 'title', 'ingredients', 'instructions'
        task_type: "generate", "complete", "substitute", etc.

    Returns:
        Dict with 'input', 'output', 'full_text'
    """
    # Your solution here
    pass


def batch_format_recipes(
    recipes: list[dict],
    template_style: str = "instruction"
) -> list[str]:
    """
    Format multiple recipes for training.

    Returns list of formatted training strings.
    """
    # Your solution here
    pass


def validate_template(template: str, tokenizer=None) -> dict:
    """
    Validate that a template is well-formed.

    Checks:
    - Has clear input/output separation
    - Special tokens are balanced
    - Not too long for context window

    Returns:
    - 'valid': bool
    - 'issues': list of problems found
    - 'token_count': if tokenizer provided
    """
    # Your solution here
    pass


def extract_response_from_template(full_text: str, template_style: str) -> str:
    """
    Extract just the response/output portion from a formatted text.

    Useful for evaluation - compare generated vs expected response.
    """
    # Your solution here
    pass


# Sample recipes for testing
SAMPLE_RECIPES = [
    {
        "title": "Simple Pasta",
        "ingredients": ["pasta", "olive oil", "garlic", "parmesan"],
        "instructions": ["Cook pasta", "Sauté garlic in oil", "Toss together", "Top with cheese"]
    },
    {
        "title": "Chocolate Cookies",
        "ingredients": ["flour", "butter", "sugar", "chocolate chips", "eggs"],
        "instructions": ["Mix dry ingredients", "Cream butter and sugar", "Combine", "Bake at 350F"]
    }
]


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: Basic instruction template
    formatted = instruction_template("Make a cake", "Here's how to make a cake...")
    assert "### Instruction:" in formatted, "Test 1a failed: missing instruction marker"
    assert "### Response:" in formatted, "Test 1b failed: missing response marker"
    assert "Make a cake" in formatted, "Test 1c failed: missing instruction text"

    # Test 2: Instruction with input
    formatted = instruction_with_input_template(
        "Substitute this ingredient",
        "butter",
        "Use coconut oil instead"
    )
    assert "### Input:" in formatted, "Test 2a failed: missing input marker"
    assert "butter" in formatted, "Test 2b failed: missing input text"

    # Test 3: Chat template
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    chat = chat_template(messages)
    assert "<|im_start|>user" in chat, "Test 3a failed: missing user marker"
    assert "<|im_end|>" in chat, "Test 3b failed: missing end marker"

    # Test 4: Chat with generation prompt
    chat_gen = chat_template(messages[:1], add_generation_prompt=True)
    assert "<|im_start|>assistant" in chat_gen, "Test 4 failed: should add assistant prompt"

    # Test 5: Recipe template
    recipe_formatted = recipe_template(
        "Test Recipe",
        ["ingredient1", "ingredient2"],
        ["step1", "step2"],
        template_style="instruction"
    )
    assert "Test Recipe" in recipe_formatted, "Test 5a failed: missing title"
    assert "ingredient1" in recipe_formatted, "Test 5b failed: missing ingredient"

    # Test 6: Different template styles
    for style in ["instruction", "chat", "structured"]:
        formatted = recipe_template("Test", ["a", "b"], ["1", "2"], template_style=style)
        assert len(formatted) > 0, f"Test 6 failed: {style} template is empty"

    # Test 7: Generation prompt
    prompt = recipe_generation_prompt(title="Chocolate Cake")
    assert "Chocolate Cake" in prompt, "Test 7a failed: missing title in prompt"

    # Test 8: Generation from ingredients
    prompt = recipe_generation_prompt(ingredients=["chicken", "rice"])
    assert "chicken" in prompt, "Test 8a failed: missing ingredient"
    assert "rice" in prompt, "Test 8b failed: missing ingredient"

    # Test 9: Training example
    example = create_training_example(SAMPLE_RECIPES[0])
    assert 'input' in example, "Test 9a failed: missing input"
    assert 'output' in example, "Test 9b failed: missing output"
    assert 'full_text' in example, "Test 9c failed: missing full_text"

    # Test 10: Batch formatting
    batch = batch_format_recipes(SAMPLE_RECIPES)
    assert len(batch) == 2, f"Test 10a failed: expected 2, got {len(batch)}"
    assert all(isinstance(s, str) for s in batch), "Test 10b failed: should be strings"

    # Test 11: Template validation
    valid_result = validate_template("### Instruction:\nTest\n\n### Response:\nOutput")
    assert 'valid' in valid_result, "Test 11a failed: missing valid key"

    # Test 12: Extract response
    full = "### Instruction:\nMake food\n\n### Response:\nHere is food"
    response = extract_response_from_template(full, "instruction")
    assert "Here is food" in response, f"Test 12 failed: got '{response}'"

    # Test 13: Llama chat template
    llama = llama_chat_template(messages)
    assert "[INST]" in llama, "Test 13a failed: missing INST tag"
    assert "[/INST]" in llama, "Test 13b failed: missing closing INST"

    # Test 14: Empty response for inference
    inference_prompt = instruction_template("Generate a recipe")
    assert "### Response:" in inference_prompt, "Test 14a failed: should have response marker"
    # Response section should be empty/minimal for inference
    response_part = inference_prompt.split("### Response:")[-1]
    assert len(response_part.strip()) < 10, "Test 14b failed: inference prompt should have empty response"

    print("All tests passed!")
    print("\nTemplate format must match between training and inference!")
    print("Choose a format and stick with it consistently.")
