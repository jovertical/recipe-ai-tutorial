# Problem 16: Substitution Explanation
#
# Generate human-readable explanations for why a substitution works.
#
# Example:
#   explainer = SubstitutionExplainer(model)
#   explanation = explainer.explain("butter", "coconut_oil", context="baking")
#   # Returns: "Coconut oil works because it provides similar fat content..."
#
# ML Relevance: Explainability builds user trust and helps identify model
# failures. This combines ML predictions with template-based generation.
#
# Your Task:
#   1. Implement feature-based explanations
#   2. Implement attention-based explanations
#   3. Implement template-based generation
#   4. Generate usage tips and warnings


from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


@dataclass
class SubstitutionExplanation:
    """Complete explanation for a substitution."""
    original: str
    substitute: str
    summary: str
    match_reasons: List[str]
    usage_tips: List[str]
    warnings: List[str]
    ratio: str
    confidence: float
    
    def to_text(self) -> str:
        """Convert to human-readable text."""
        lines = [f"Substitute {self.original} with {self.substitute}"]
        lines.append(f"\n{self.summary}")
        if self.match_reasons:
            lines.append("\nWhy it works:")
            for reason in self.match_reasons:
                lines.append(f"  - {reason}")
        if self.usage_tips:
            lines.append("\nTips:")
            for tip in self.usage_tips:
                lines.append(f"  - {tip}")
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        lines.append(f"\nRatio: {self.ratio}")
        lines.append(f"Confidence: {self.confidence:.0%}")
        return "\n".join(lines)


class ExplanationTemplate:
    """Template-based explanation generation."""
    
    # Summary templates
    SUMMARY_TEMPLATES = {
        "perfect_match": "{substitute} is an excellent substitute for {original} because they share similar {properties}.",
        "good_match": "{substitute} works well as a substitute for {original}, providing {functions}.",
        "partial_match": "{substitute} can replace {original} for {functions}, though some adjustments may be needed.",
        "context_specific": "In {context}, {substitute} can replace {original} because {reason}.",
    }
    
    # Reason templates
    REASON_TEMPLATES = {
        "same_category": "Both are {category} ingredients",
        "same_function": "Both provide {function} in recipes",
        "similar_texture": "Similar texture ({texture})",
        "similar_flavor": "Compatible flavor profiles",
        "dietary_benefit": "Suitable for {diet} diets",
    }
    
    # Warning templates
    WARNING_TEMPLATES = {
        "flavor_change": "May alter the flavor of the dish",
        "texture_change": "May affect texture, especially in {context}",
        "cooking_behavior": "Behaves differently when {cooking_method}",
        "allergen": "Contains {allergen} - not suitable for those with allergies",
        "dietary": "Not suitable for {diet} diets",
    }
    
    def generate_summary(
        self,
        original: str,
        substitute: str,
        match_quality: str,
        properties: List[str] = None,
        context: str = None
    ) -> str:
        """Generate summary from template."""
        # Your solution here
        pass
    
    def generate_reasons(
        self,
        match_info: Dict[str, any]
    ) -> List[str]:
        """Generate match reasons from template."""
        # Your solution here
        pass
    
    def generate_warnings(
        self,
        warning_info: Dict[str, any]
    ) -> List[str]:
        """Generate warnings from template."""
        # Your solution here
        pass


class ExplanationGenerator:
    """Generate explanations for substitutions."""
    
    def __init__(
        self,
        ingredient_db: Dict[str, Dict],
        function_db: Dict[str, Set[str]] = None,
        template_engine: ExplanationTemplate = None
    ):
        """
        Initialize generator.
        
        Args:
            ingredient_db: Ingredient properties database
            function_db: Ingredient -> functions mapping
            template_engine: Template engine for text generation
        """
        self.ingredient_db = ingredient_db
        self.function_db = function_db or {}
        self.templates = template_engine or ExplanationTemplate()
    
    def explain(
        self,
        original: str,
        substitute: str,
        context: str = None,
        score: float = None
    ) -> SubstitutionExplanation:
        """
        Generate full explanation for a substitution.
        
        Args:
            original: Original ingredient
            substitute: Proposed substitute
            context: Recipe context
            score: Substitution score
            
        Returns:
            Complete explanation
        """
        # Your solution here
        pass
    
    def get_match_reasons(
        self,
        original: str,
        substitute: str,
        context: str = None
    ) -> List[str]:
        """Generate reasons why substitution works."""
        # Your solution here
        pass
    
    def get_usage_tips(
        self,
        original: str,
        substitute: str,
        context: str = None
    ) -> List[str]:
        """Generate usage tips for the substitution."""
        # Your solution here
        pass
    
    def get_warnings(
        self,
        original: str,
        substitute: str,
        context: str = None
    ) -> List[str]:
        """Generate warnings about the substitution."""
        # Your solution here
        pass
    
    def get_ratio(
        self,
        original: str,
        substitute: str,
        context: str = None
    ) -> str:
        """Get substitution ratio."""
        # Your solution here
        pass


class FeatureExplainer:
    """Explain based on feature comparison."""
    
    def __init__(
        self,
        feature_names: List[str],
        feature_importance: Dict[str, float] = None
    ):
        """
        Initialize explainer.
        
        Args:
            feature_names: Names of features used in matching
            feature_importance: Importance weight for each feature
        """
        self.feature_names = feature_names
        self.importance = feature_importance or {}
    
    def explain_match(
        self,
        original_features: np.ndarray,
        substitute_features: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Explain which features matched.
        
        Returns:
            List of (feature_name, description, match_score)
        """
        # Your solution here
        pass
    
    def explain_difference(
        self,
        original_features: np.ndarray,
        substitute_features: np.ndarray,
        threshold: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Explain key differences between ingredients.
        
        Returns:
            List of (feature_name, description, difference)
        """
        # Your solution here
        pass
    
    def generate_feature_summary(
        self,
        matches: List[Tuple[str, str, float]],
        differences: List[Tuple[str, str, float]]
    ) -> str:
        """Generate text summary from feature analysis."""
        # Your solution here
        pass


class TipGenerator:
    """Generate usage tips for substitutions."""
    
    # Tip templates by category
    TIP_TEMPLATES = {
        "ratio": "Use {ratio} {substitute} for every {original_amount} {original}",
        "texture": "For best texture, {tip}",
        "temperature": "{substitute} works best when {temperature}",
        "timing": "Add {substitute} {timing} in the recipe",
        "combination": "Combine with {companion} for better results",
        "adjustment": "You may need to {adjustment}",
    }
    
    def __init__(self, tip_database: Dict[Tuple[str, str], List[str]] = None):
        """
        Initialize generator.
        
        Args:
            tip_database: (original, substitute) -> tips mapping
        """
        self.tip_db = tip_database or {}
    
    def get_tips(
        self,
        original: str,
        substitute: str,
        context: str = None,
        max_tips: int = 5
    ) -> List[str]:
        """Get usage tips for substitution."""
        # Your solution here
        pass
    
    def generate_ratio_tip(
        self,
        original: str,
        substitute: str
    ) -> str:
        """Generate ratio tip."""
        # Your solution here
        pass
    
    def generate_adjustment_tips(
        self,
        original: str,
        substitute: str,
        context: str = None
    ) -> List[str]:
        """Generate adjustment tips based on differences."""
        # Your solution here
        pass


class WarningGenerator:
    """Generate warnings for substitutions."""
    
    WARNING_TYPES = {
        "allergen": "⚠ Contains {allergen}",
        "dietary": "⚠ Not suitable for {diet} diet",
        "flavor": "⚠ May change flavor: {description}",
        "texture": "⚠ May affect texture: {description}",
        "cooking": "⚠ Different cooking behavior: {description}",
        "nutrition": "⚠ Nutritional difference: {description}",
    }
    
    def __init__(
        self,
        allergen_db: Dict[str, Set[str]] = None,
        dietary_db: Dict[str, Dict[str, bool]] = None
    ):
        """Initialize warning generator."""
        self.allergen_db = allergen_db or {}
        self.dietary_db = dietary_db or {}
    
    def get_warnings(
        self,
        original: str,
        substitute: str,
        user_restrictions: List[str] = None,
        context: str = None
    ) -> List[str]:
        """Get all relevant warnings."""
        # Your solution here
        pass
    
    def check_allergens(
        self,
        ingredient: str,
        user_allergens: List[str] = None
    ) -> List[str]:
        """Check for allergen warnings."""
        # Your solution here
        pass
    
    def check_dietary(
        self,
        ingredient: str,
        required_diets: List[str] = None
    ) -> List[str]:
        """Check for dietary warnings."""
        # Your solution here
        pass


def explain_batch(
    generator: ExplanationGenerator,
    substitutions: List[Tuple[str, str]],
    context: str = None
) -> List[SubstitutionExplanation]:
    """Generate explanations for multiple substitutions."""
    # Your solution here
    pass


def format_explanation_html(
    explanation: SubstitutionExplanation
) -> str:
    """Format explanation as HTML."""
    # Your solution here
    pass


def format_explanation_markdown(
    explanation: SubstitutionExplanation
) -> str:
    """Format explanation as Markdown."""
    # Your solution here
    pass


# Sample databases
SAMPLE_INGREDIENT_DB = {
    "butter": {
        "category": "fat",
        "texture": "solid",
        "flavor": "rich, creamy",
        "dietary": ["vegetarian"],
        "allergens": ["milk"],
    },
    "coconut_oil": {
        "category": "fat",
        "texture": "solid",
        "flavor": "mild coconut",
        "dietary": ["vegan", "dairy_free"],
        "allergens": [],
    },
    "olive_oil": {
        "category": "fat",
        "texture": "liquid",
        "flavor": "fruity, peppery",
        "dietary": ["vegan", "dairy_free"],
        "allergens": [],
    },
}

SAMPLE_RATIO_DB = {
    ("butter", "coconut_oil"): "1:1",
    ("butter", "olive_oil"): "3/4 cup olive oil per 1 cup butter",
    ("butter", "applesauce"): "1/2 cup applesauce per 1 cup butter",
    ("egg", "flax_egg"): "1 flax egg per 1 egg",
}


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing ExplanationTemplate...")
    
    template = ExplanationTemplate()
    
    # Test 1: Generate summary
    summary = template.generate_summary(
        "butter", "coconut_oil",
        match_quality="good_match",
        properties=["fat", "solid texture"]
    )
    assert len(summary) > 0, "Test 1 failed"
    print(f"  ✓ Summary: {summary}")
    
    # Test 2: Generate reasons
    match_info = {
        "same_category": "fat",
        "same_function": ["moisture", "tenderness"],
        "similar_texture": "solid"
    }
    reasons = template.generate_reasons(match_info)
    assert len(reasons) > 0, "Test 2 failed"
    print(f"  ✓ Reasons: {reasons}")
    
    # Test 3: Generate warnings
    warning_info = {
        "flavor_change": True,
        "context": "baking"
    }
    warnings = template.generate_warnings(warning_info)
    print(f"  ✓ Warnings: {warnings}")
    
    print("\nTesting ExplanationGenerator...")
    
    function_db = {
        "butter": {"fat", "moisture", "flavor"},
        "coconut_oil": {"fat", "moisture"},
    }
    generator = ExplanationGenerator(SAMPLE_INGREDIENT_DB, function_db, template)
    
    # Test 4: Full explanation
    explanation = generator.explain("butter", "coconut_oil", context="baking")
    assert explanation is not None, "Test 4a failed"
    assert len(explanation.summary) > 0, "Test 4b failed"
    print(f"  ✓ Generated explanation")
    print(f"    Summary: {explanation.summary}")
    
    # Test 5: Match reasons
    reasons = generator.get_match_reasons("butter", "coconut_oil")
    assert len(reasons) > 0, "Test 5 failed"
    print(f"  ✓ Match reasons: {reasons}")
    
    # Test 6: Usage tips
    tips = generator.get_usage_tips("butter", "coconut_oil", "baking")
    print(f"  ✓ Usage tips: {tips}")
    
    # Test 7: Warnings
    warnings = generator.get_warnings("butter", "coconut_oil")
    print(f"  ✓ Warnings: {warnings}")
    
    # Test 8: Ratio
    ratio = generator.get_ratio("butter", "coconut_oil")
    assert len(ratio) > 0, "Test 8 failed"
    print(f"  ✓ Ratio: {ratio}")
    
    print("\nTesting FeatureExplainer...")
    
    feature_names = ["sweetness", "saltiness", "fattiness", "moisture"]
    explainer = FeatureExplainer(feature_names)
    
    # Test 9: Explain match
    original_feat = np.array([0.1, 0.2, 0.9, 0.8])
    substitute_feat = np.array([0.2, 0.2, 0.85, 0.75])
    matches = explainer.explain_match(original_feat, substitute_feat, top_k=2)
    assert len(matches) <= 2, "Test 9 failed"
    print(f"  ✓ Feature matches: {matches}")
    
    # Test 10: Explain differences
    differences = explainer.explain_difference(original_feat, substitute_feat)
    print(f"  ✓ Feature differences: {differences}")
    
    print("\nTesting TipGenerator...")
    
    tip_gen = TipGenerator()
    
    # Test 11: Get tips
    tips = tip_gen.get_tips("butter", "coconut_oil", "baking")
    print(f"  ✓ Tips: {tips}")
    
    # Test 12: Ratio tip
    ratio_tip = tip_gen.generate_ratio_tip("butter", "coconut_oil")
    assert len(ratio_tip) > 0, "Test 12 failed"
    print(f"  ✓ Ratio tip: {ratio_tip}")
    
    print("\nTesting WarningGenerator...")
    
    allergen_db = {"butter": {"milk"}, "coconut_oil": set()}
    dietary_db = {
        "butter": {"vegan": False, "vegetarian": True},
        "coconut_oil": {"vegan": True, "vegetarian": True}
    }
    warning_gen = WarningGenerator(allergen_db, dietary_db)
    
    # Test 13: Get warnings
    warnings = warning_gen.get_warnings(
        "butter", "coconut_oil",
        user_restrictions=["vegan"]
    )
    print(f"  ✓ Warnings: {warnings}")
    
    # Test 14: Check allergens
    allergen_warnings = warning_gen.check_allergens("butter", ["milk"])
    assert len(allergen_warnings) > 0, "Test 14 failed"
    print(f"  ✓ Allergen warnings: {allergen_warnings}")
    
    print("\nTesting full explanation output...")
    
    # Test 15: To text
    explanation = generator.explain("butter", "coconut_oil", "baking", 0.85)
    text = explanation.to_text()
    assert len(text) > 0, "Test 15 failed"
    print(f"\n{text}")
    
    print("\nTesting batch explanation...")
    
    # Test 16: Batch
    subs = [("butter", "coconut_oil"), ("butter", "olive_oil")]
    explanations = explain_batch(generator, subs, "baking")
    assert len(explanations) == 2, "Test 16 failed"
    print(f"  ✓ Generated {len(explanations)} explanations")
    
    print("\nTesting formatting...")
    
    # Test 17: Markdown format
    md = format_explanation_markdown(explanation)
    assert "##" in md or "#" in md or len(md) > 0, "Test 17 failed"
    print(f"  ✓ Markdown generated ({len(md)} chars)")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Template-based generation is interpretable")
    print("2. Feature attribution explains matching")
    print("3. Tips improve user experience")
    print("4. Warnings ensure safety")
    print("\nNext: Confidence scoring and uncertainty!")
