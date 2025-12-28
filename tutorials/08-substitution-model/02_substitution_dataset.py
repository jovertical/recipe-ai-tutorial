# Problem 2: Building a Substitution Dataset
#
# Learn how to collect and structure training data for ingredient
# substitution models. Good data is the foundation of ML.
#
# Example:
#   # Positive pairs (valid substitutions)
#   ("butter", "margarine", 1)
#   ("milk", "oat_milk", 1)
#
#   # Negative pairs (invalid substitutions)
#   ("butter", "sugar", 0)
#   ("salt", "cinnamon", 0)
#
# ML Relevance: Training substitution models requires:
# - Positive examples: known valid substitutions
# - Negative examples: pairs that don't substitute
# - Context information: when substitutions work
# - Quality labels: how good each substitution is
#
# Data quality directly impacts model performance.

from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import random
import json
from collections import defaultdict


@dataclass
class SubstitutionPair:
    """A single substitution example."""
    original: str
    substitute: str
    label: float  # 1.0 = perfect substitute, 0.0 = not a substitute
    context: Optional[str] = None  # e.g., "baking", "frying"
    notes: Optional[str] = None
    source: Optional[str] = None

    def __repr__(self):
        return f"{self.original} -> {self.substitute} ({self.label:.1f})"


# Sample substitution knowledge base
SUBSTITUTION_KB = {
    "butter": {
        "substitutes": ["margarine", "coconut_oil", "vegetable_oil", "applesauce", "greek_yogurt"],
        "context": {
            "baking": ["margarine", "coconut_oil", "applesauce"],
            "frying": ["vegetable_oil", "coconut_oil", "lard"],
            "spreading": ["margarine", "cream_cheese", "avocado"]
        }
    },
    "egg": {
        "substitutes": ["flax_egg", "chia_egg", "applesauce", "banana", "aquafaba"],
        "context": {
            "binding": ["flax_egg", "chia_egg"],
            "leavening": ["aquafaba", "baking_powder_mix"],
            "moisture": ["applesauce", "banana", "pumpkin_puree"]
        }
    },
    "milk": {
        "substitutes": ["oat_milk", "almond_milk", "soy_milk", "coconut_milk", "cashew_milk"],
        "context": {
            "drinking": ["oat_milk", "almond_milk", "soy_milk"],
            "baking": ["any"],
            "cream_sauce": ["coconut_milk", "cashew_cream"]
        }
    },
    "flour": {
        "substitutes": ["almond_flour", "oat_flour", "coconut_flour", "rice_flour"],
        "context": {
            "baking": ["almond_flour", "oat_flour"],
            "thickening": ["rice_flour", "cornstarch", "arrowroot"]
        }
    },
    "sugar": {
        "substitutes": ["honey", "maple_syrup", "coconut_sugar", "stevia", "monk_fruit"],
        "context": {
            "baking": ["coconut_sugar", "honey", "maple_syrup"],
            "sweetening": ["stevia", "monk_fruit", "erythritol"]
        }
    },
    "cream": {
        "substitutes": ["coconut_cream", "cashew_cream", "silken_tofu", "evaporated_milk"],
        "context": {
            "whipping": ["coconut_cream"],
            "sauce": ["cashew_cream", "silken_tofu"],
            "coffee": ["oat_creamer", "coconut_creamer"]
        }
    },
    "sour_cream": {
        "substitutes": ["greek_yogurt", "coconut_cream", "cashew_cream", "silken_tofu"],
        "context": {
            "topping": ["greek_yogurt", "coconut_yogurt"],
            "baking": ["greek_yogurt", "buttermilk"]
        }
    },
    "beef": {
        "substitutes": ["ground_turkey", "ground_chicken", "tempeh", "seitan", "mushrooms"],
        "context": {
            "burger": ["ground_turkey", "black_bean_patty", "impossible_meat"],
            "stew": ["seitan", "jackfruit", "mushrooms"]
        }
    },
    "chicken": {
        "substitutes": ["tofu", "tempeh", "seitan", "jackfruit", "cauliflower"],
        "context": {
            "stir_fry": ["tofu", "tempeh"],
            "pulled": ["jackfruit"],
            "fried": ["cauliflower", "tofu"]
        }
    },
    "pasta": {
        "substitutes": ["zucchini_noodles", "spaghetti_squash", "rice_noodles", "shirataki"],
        "context": {
            "low_carb": ["zucchini_noodles", "shirataki"],
            "gluten_free": ["rice_noodles", "chickpea_pasta"]
        }
    }
}

# All known ingredients for negative sampling
ALL_INGREDIENTS = [
    "butter", "margarine", "coconut_oil", "vegetable_oil", "olive_oil", "lard",
    "egg", "flax_egg", "chia_egg", "aquafaba",
    "milk", "oat_milk", "almond_milk", "soy_milk", "coconut_milk",
    "flour", "almond_flour", "oat_flour", "coconut_flour", "rice_flour",
    "sugar", "honey", "maple_syrup", "coconut_sugar", "stevia",
    "cream", "coconut_cream", "cashew_cream",
    "chicken", "beef", "pork", "tofu", "tempeh", "seitan",
    "salt", "pepper", "garlic", "onion", "ginger",
    "tomato", "lettuce", "carrot", "celery", "broccoli",
    "apple", "banana", "orange", "lemon", "lime",
    "rice", "pasta", "bread", "quinoa", "oats",
    "cheese", "parmesan", "cheddar", "mozzarella",
    "water", "broth", "wine", "vinegar"
]


def load_substitution_pairs(
    source: str = "kb",
    include_context: bool = True
) -> List[SubstitutionPair]:
    """Load substitution pairs from a data source."""
    # Your solution here
    pass


def get_valid_substitutes(ingredient: str) -> Set[str]:
    """Get all valid substitutes for an ingredient."""
    # Your solution here
    pass


def generate_negative_samples(
    positive_pairs: List[SubstitutionPair],
    num_negatives_per_positive: int = 3,
    strategy: str = "random"
) -> List[SubstitutionPair]:
    """
    Generate negative (non-substitute) pairs.

    Args:
        positive_pairs: List of valid substitution pairs
        num_negatives_per_positive: How many negatives per positive
        strategy: Sampling strategy
            - "random": Random non-substitute pairs
            - "same_category": Hard negatives from same category
            - "mixed": Mix of random and hard negatives

    Returns:
        List of negative SubstitutionPair (label=0)
    """
    # Your solution here
    pass


def create_training_split(
    pairs: List[SubstitutionPair],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str = "ingredient"
) -> Tuple[List[SubstitutionPair], List[SubstitutionPair], List[SubstitutionPair]]:
    """Split data into train/validation/test sets."""
    # Your solution here
    pass


def balance_dataset(
    pairs: List[SubstitutionPair],
    target_ratio: float = 1.0
) -> List[SubstitutionPair]:
    """Balance positive and negative examples."""
    # Your solution here
    pass


def augment_pairs(
    pairs: List[SubstitutionPair],
    include_reverse: bool = True,
    include_synonyms: bool = False
) -> List[SubstitutionPair]:
    """Augment substitution pairs."""
    # Your solution here
    pass


class SubstitutionDataset:
    """Dataset class for substitution model training."""

    def __init__(
        self,
        pairs: List[SubstitutionPair],
        vocabulary: Dict[str, int] = None
    ):
        """Initialize dataset."""
        # Your solution here
        pass

    def __len__(self) -> int:
        """Return number of pairs."""
        # Your solution here
        pass

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Get a single example."""
        # Your solution here
        pass

    def get_batch(self, indices: List[int]) -> Dict[str, List]:
        """Get a batch of examples."""
        # Your solution here
        pass

    def shuffle(self):
        """Shuffle the dataset in place."""
        # Your solution here
        pass

    def iterate_batches(self, batch_size: int, shuffle: bool = True):
        """Iterate over batches."""
        # Your solution here
        pass

    def get_statistics(self) -> Dict[str, any]:
        """Compute dataset statistics."""
        # Your solution here
        pass

    def build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary from ingredients in dataset."""
        # Your solution here
        pass


def save_dataset(
    pairs: List[SubstitutionPair],
    filepath: str,
    format: str = "jsonl"
):
    """Save dataset to file."""
    # Your solution here
    pass


def load_dataset(filepath: str) -> List[SubstitutionPair]:
    """Load dataset from file."""
    # Your solution here
    pass


def analyze_dataset(pairs: List[SubstitutionPair]) -> Dict[str, any]:
    """Analyze dataset for quality and coverage."""
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    random.seed(42)

    print("Testing load_substitution_pairs...")

    # Test 1: Load from knowledge base
    pairs = load_substitution_pairs("kb", include_context=False)
    assert len(pairs) > 0, "Test 1a failed: should load pairs"
    assert all(isinstance(p, SubstitutionPair) for p in pairs), "Test 1b failed"
    assert all(p.label == 1.0 for p in pairs), "Test 1c failed: KB pairs are positive"
    print(f"  ✓ Loaded {len(pairs)} pairs from KB")

    # Test 2: With context
    pairs_with_context = load_substitution_pairs("kb", include_context=True)
    assert len(pairs_with_context) >= len(pairs), "Test 2 failed: context should add pairs"
    print(f"  ✓ Loaded {len(pairs_with_context)} pairs with context")

    print("\nTesting get_valid_substitutes...")

    # Test 3: Get substitutes
    subs = get_valid_substitutes("butter")
    assert "margarine" in subs, "Test 3a failed"
    assert "coconut_oil" in subs, "Test 3b failed"
    assert "butter" not in subs, "Test 3c failed: ingredient shouldn't substitute itself"
    print(f"  ✓ Found {len(subs)} substitutes for butter")

    # Test 4: Unknown ingredient
    subs = get_valid_substitutes("unicorn_tears")
    assert len(subs) == 0, "Test 4 failed"
    print("  ✓ Unknown ingredient returns empty set")

    print("\nTesting generate_negative_samples...")

    # Test 5: Generate negatives
    positives = load_substitution_pairs("kb", include_context=False)
    negatives = generate_negative_samples(positives, num_negatives_per_positive=2)
    assert len(negatives) > 0, "Test 5a failed"
    assert all(n.label == 0.0 for n in negatives), "Test 5b failed: negatives should have label 0"
    print(f"  ✓ Generated {len(negatives)} negative samples")

    # Test 6: Negatives are not valid substitutes
    valid_pairs = {(p.original, p.substitute) for p in positives}
    valid_pairs.update({(p.substitute, p.original) for p in positives})  # Include reverse
    invalid_negatives = 0
    for n in negatives:
        if (n.original, n.substitute) in valid_pairs:
            invalid_negatives += 1
    assert invalid_negatives == 0, f"Test 6 failed: {invalid_negatives} negatives are actually valid"
    print("  ✓ All negatives are actually non-substitutes")

    print("\nTesting create_training_split...")

    # Test 7: Basic split
    all_pairs = positives + negatives
    train, val, test = create_training_split(all_pairs, 0.7, 0.15, 0.15)
    total = len(train) + len(val) + len(test)
    assert total == len(all_pairs), f"Test 7a failed: {total} vs {len(all_pairs)}"
    assert abs(len(train) / len(all_pairs) - 0.7) < 0.1, "Test 7b failed: train ratio off"
    print(f"  ✓ Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Test 8: No overlap
    train_set = {(p.original, p.substitute) for p in train}
    val_set = {(p.original, p.substitute) for p in val}
    test_set = {(p.original, p.substitute) for p in test}
    assert len(train_set & val_set) == 0, "Test 8a failed: train/val overlap"
    assert len(train_set & test_set) == 0, "Test 8b failed: train/test overlap"
    print("  ✓ No overlap between splits")

    print("\nTesting balance_dataset...")

    # Test 9: Balance
    imbalanced = positives + negatives + negatives  # 1:2 ratio
    balanced = balance_dataset(imbalanced, target_ratio=1.0)
    pos_count = sum(1 for p in balanced if p.label == 1.0)
    neg_count = sum(1 for p in balanced if p.label == 0.0)
    assert abs(pos_count - neg_count) <= 1, f"Test 9 failed: {pos_count} vs {neg_count}"
    print(f"  ✓ Balanced to {pos_count} positive, {neg_count} negative")

    print("\nTesting augment_pairs...")

    # Test 10: Augment with reverse
    small_pairs = [SubstitutionPair("butter", "margarine", 1.0)]
    augmented = augment_pairs(small_pairs, include_reverse=True)
    assert len(augmented) == 2, f"Test 10 failed: {len(augmented)}"
    reverse_exists = any(p.original == "margarine" and p.substitute == "butter" for p in augmented)
    assert reverse_exists, "Test 10b failed: reverse not found"
    print("  ✓ Augmentation with reverse works")

    print("\nTesting SubstitutionDataset...")

    # Test 11: Dataset initialization
    dataset = SubstitutionDataset(all_pairs)
    assert len(dataset) == len(all_pairs), f"Test 11 failed: {len(dataset)}"
    print(f"  ✓ Dataset created with {len(dataset)} pairs")

    # Test 12: Get item
    item = dataset[0]
    assert "original" in item, "Test 12a failed"
    assert "substitute" in item, "Test 12b failed"
    assert "label" in item, "Test 12c failed"
    print(f"  ✓ Item: {item['original']} -> {item['substitute']}")

    # Test 13: Get batch
    batch = dataset.get_batch([0, 1, 2])
    assert len(batch["original"]) == 3, "Test 13 failed"
    print("  ✓ Batch retrieval works")

    # Test 14: Statistics
    stats = dataset.get_statistics()
    assert "num_pairs" in stats, "Test 14a failed"
    assert "positive_ratio" in stats, "Test 14b failed"
    assert stats["num_pairs"] == len(all_pairs), "Test 14c failed"
    print(f"  ✓ Stats: {stats['num_positive']} pos, {stats['num_negative']} neg")

    # Test 15: Build vocabulary
    vocab = dataset.build_vocabulary()
    assert len(vocab) > 0, "Test 15a failed"
    assert all(isinstance(v, int) for v in vocab.values()), "Test 15b failed"
    print(f"  ✓ Built vocabulary with {len(vocab)} ingredients")

    # Test 16: Iterate batches
    batch_count = 0
    for batch in dataset.iterate_batches(batch_size=10, shuffle=False):
        batch_count += 1
    expected_batches = (len(all_pairs) + 9) // 10
    assert batch_count == expected_batches, f"Test 16 failed: {batch_count} vs {expected_batches}"
    print(f"  ✓ Batch iteration: {batch_count} batches of size 10")

    print("\nTesting analyze_dataset...")

    # Test 17: Analysis
    analysis = analyze_dataset(all_pairs)
    assert "ingredient_coverage" in analysis, "Test 17a failed"
    assert "label_distribution" in analysis, "Test 17b failed"
    print(f"  ✓ Analysis: {analysis.get('ingredient_coverage', 0)} ingredients covered")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)

    print("\nDataset Summary:")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Positive: {sum(1 for p in all_pairs if p.label == 1.0)}")
    print(f"  Negative: {sum(1 for p in all_pairs if p.label == 0.0)}")
    print(f"  Unique ingredients: {len(vocab)}")

    print("\nKey concepts learned:")
    print("1. Positive pairs from knowledge bases or crowdsourcing")
    print("2. Negative sampling creates training signal")
    print("3. Proper splits prevent data leakage")
    print("4. Balanced datasets improve training")
