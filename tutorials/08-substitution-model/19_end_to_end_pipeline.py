# Problem 19: End-to-End Pipeline
#
# Build a complete substitution pipeline from raw input to final recommendations.
#
# Example:
#   pipeline = SubstitutionPipeline(retriever, reranker, explainer)
#   results = pipeline.run("butter", recipe_context, constraints)
#   # Returns full substitution results with scores and explanations
#
# ML Relevance: Production systems chain multiple models together - retrieval,
# ranking, filtering, and explanation generation.
#
# Your Task:
#   1. Implement pipeline orchestration
#   2. Implement stage-wise processing
#   3. Implement result aggregation
#   4. Implement end-to-end evaluation


from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import time
import numpy as np
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the substitution pipeline."""
    
    # Model paths
    embedding_model_path: str = None
    ranking_model_path: str = None
    
    # Retrieval settings
    retrieval_top_k: int = 50
    rerank_top_k: int = 10
    final_top_k: int = 5
    
    # Feature weights
    embedding_weight: float = 0.3
    flavor_weight: float = 0.2
    texture_weight: float = 0.2
    function_weight: float = 0.2
    dietary_weight: float = 0.1
    
    # Options
    use_context: bool = True
    use_reranking: bool = True
    generate_explanations: bool = True
    estimate_confidence: bool = True
    
    # Cache settings
    enable_cache: bool = True
    cache_size: int = 1000
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'PipelineConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: str) -> 'PipelineConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class SubstitutionResult:
    """Complete result from the pipeline."""
    original: str
    substitutes: List[Dict]
    recipe_context: List[str]
    dietary_requirements: List[str]
    
    # Metadata
    processing_time_ms: float = 0
    pipeline_version: str = "1.0.0"
    
    # Component outputs
    retrieval_candidates: int = 0
    after_dietary_filter: int = 0
    after_reranking: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_top_substitute(self) -> Optional[Dict]:
        """Get the best substitute."""
        return self.substitutes[0] if self.substitutes else None
    
    def summary(self) -> str:
        """Generate text summary."""
        if not self.substitutes:
            return f"No substitutes found for {self.original}"
        
        lines = [f"Substitutes for {self.original}:"]
        for i, sub in enumerate(self.substitutes[:3], 1):
            lines.append(f"  {i}. {sub['name']} (score: {sub['score']:.2f})")
        return "\n".join(lines)


class PipelineStage(Enum):
    """Stages in the pipeline."""
    PREPROCESS = "preprocess"
    RETRIEVE = "retrieve"
    FILTER = "filter"
    RANK = "rank"
    EXPLAIN = "explain"
    POSTPROCESS = "postprocess"


class SubstitutionPipeline:
    """
    End-to-end pipeline for ingredient substitution.
    
    Integrates:
    - Embedding retrieval
    - Dietary constraint filtering
    - Context-aware ranking
    - Flavor/texture matching
    - Explanation generation
    - Confidence estimation
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        embedding_retriever=None,
        dietary_checker=None,
        context_scorer=None,
        flavor_matcher=None,
        texture_matcher=None,
        reranker=None,
        explainer=None,
        confidence_scorer=None
    ):
        """
        Initialize pipeline with components.
        
        Args:
            config: Pipeline configuration
            embedding_retriever: For initial candidate retrieval
            dietary_checker: For dietary constraint filtering
            context_scorer: For context-aware scoring
            flavor_matcher: For flavor profile matching
            texture_matcher: For texture matching
            reranker: For re-ranking candidates
            explainer: For generating explanations
            confidence_scorer: For confidence estimation
        """
        self.config = config
        self.retriever = embedding_retriever
        self.dietary_checker = dietary_checker
        self.context_scorer = context_scorer
        self.flavor_matcher = flavor_matcher
        self.texture_matcher = texture_matcher
        self.reranker = reranker
        self.explainer = explainer
        self.confidence_scorer = confidence_scorer
        
        self.version = "1.0.0"
        self.stage_times = {}
    
    @classmethod
    def from_config(cls, config: PipelineConfig) -> 'SubstitutionPipeline':
        """
        Create pipeline from configuration.
        
        Loads models and initializes components.
        """
        # Your solution here
        # Load models based on config paths
        # Initialize all components
        pass
    
    def run(
        self,
        ingredient: str,
        recipe: List[str] = None,
        dietary_requirements: List[str] = None,
        allergens_to_avoid: List[str] = None,
        context: str = None,
        cooking_method: str = None
    ) -> SubstitutionResult:
        """
        Run the full substitution pipeline.
        
        Args:
            ingredient: Ingredient to substitute
            recipe: Other ingredients in the recipe
            dietary_requirements: Dietary restrictions
            allergens_to_avoid: Allergens to avoid
            context: Recipe context description
            cooking_method: Cooking method being used
            
        Returns:
            Complete substitution result
        """
        start_time = time.time()
        self.stage_times = {}
        
        # Stage 1: Preprocess
        preprocessed = self._preprocess(
            ingredient, recipe, dietary_requirements,
            allergens_to_avoid, context, cooking_method
        )
        
        # Stage 2: Retrieve candidates
        candidates = self._retrieve(preprocessed)
        retrieval_count = len(candidates)
        
        # Stage 3: Filter by constraints
        filtered = self._filter(candidates, preprocessed)
        filter_count = len(filtered)
        
        # Stage 4: Rank candidates
        ranked = self._rank(filtered, preprocessed)
        
        # Stage 5: Generate explanations
        if self.config.generate_explanations:
            ranked = self._explain(ranked, preprocessed)
        
        # Stage 6: Estimate confidence
        if self.config.estimate_confidence:
            ranked = self._add_confidence(ranked, preprocessed)
        
        # Stage 7: Postprocess
        final = self._postprocess(ranked, preprocessed)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SubstitutionResult(
            original=ingredient,
            substitutes=final[:self.config.final_top_k],
            recipe_context=recipe or [],
            dietary_requirements=dietary_requirements or [],
            processing_time_ms=processing_time,
            pipeline_version=self.version,
            retrieval_candidates=retrieval_count,
            after_dietary_filter=filter_count,
            after_reranking=len(final)
        )
    
    def _preprocess(
        self,
        ingredient: str,
        recipe: List[str],
        dietary: List[str],
        allergens: List[str],
        context: str,
        cooking_method: str
    ) -> Dict:
        """Preprocess inputs."""
        stage_start = time.time()
        
        result = {
            "ingredient": ingredient.lower().strip(),
            "recipe": [r.lower().strip() for r in (recipe or [])],
            "dietary": dietary or [],
            "allergens": allergens or [],
            "context": context,
            "cooking_method": cooking_method
        }
        
        self.stage_times["preprocess"] = time.time() - stage_start
        return result
    
    def _retrieve(self, inputs: Dict) -> List[Dict]:
        """Retrieve initial candidates."""
        stage_start = time.time()
        
        candidates = []
        if self.retriever:
            raw_candidates = self.retriever.find_similar(
                inputs["ingredient"],
                top_k=self.config.retrieval_top_k
            )
            candidates = [
                {"name": name, "embedding_score": score}
                for name, score in raw_candidates
            ]
        
        self.stage_times["retrieve"] = time.time() - stage_start
        return candidates
    
    def _filter(self, candidates: List[Dict], inputs: Dict) -> List[Dict]:
        """Filter by constraints."""
        stage_start = time.time()
        
        filtered = []
        for c in candidates:
            # Check dietary
            if self.dietary_checker and inputs["dietary"]:
                if not self.dietary_checker.check(c["name"], inputs["dietary"]):
                    continue
            
            # Check allergens
            if self.dietary_checker and inputs["allergens"]:
                if not self.dietary_checker.is_safe(c["name"], inputs["allergens"]):
                    continue
            
            filtered.append(c)
        
        self.stage_times["filter"] = time.time() - stage_start
        return filtered
    
    def _rank(self, candidates: List[Dict], inputs: Dict) -> List[Dict]:
        """Rank candidates."""
        stage_start = time.time()
        
        for c in candidates:
            scores = {}
            
            # Embedding score (already have it)
            scores["embedding"] = c.get("embedding_score", 0)
            
            # Flavor score
            if self.flavor_matcher:
                scores["flavor"] = self.flavor_matcher.score(
                    inputs["ingredient"], c["name"]
                )
            
            # Texture score
            if self.texture_matcher:
                scores["texture"] = self.texture_matcher.score(
                    inputs["ingredient"], c["name"],
                    cooking_method=inputs["cooking_method"]
                )
            
            # Context score
            if self.context_scorer and inputs["context"]:
                scores["context"] = self.context_scorer.score(
                    c["name"], inputs["context"]
                )
            
            # Combined score
            c["scores"] = scores
            c["score"] = self._combine_scores(scores)
        
        # Re-rank with cross-encoder if available
        if self.config.use_reranking and self.reranker:
            candidates = candidates[:self.config.rerank_top_k]
            for c in candidates:
                c["rerank_score"] = self.reranker.score(
                    inputs["ingredient"], c["name"]
                )
                c["score"] = 0.3 * c["score"] + 0.7 * c["rerank_score"]
        
        # Sort by final score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        self.stage_times["rank"] = time.time() - stage_start
        return candidates
    
    def _combine_scores(self, scores: Dict[str, float]) -> float:
        """Combine individual scores."""
        weights = {
            "embedding": self.config.embedding_weight,
            "flavor": self.config.flavor_weight,
            "texture": self.config.texture_weight,
            "context": self.config.function_weight,
        }
        
        total_weight = 0
        weighted_sum = 0
        for name, score in scores.items():
            weight = weights.get(name, 0.1)
            weighted_sum += weight * score
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _explain(self, candidates: List[Dict], inputs: Dict) -> List[Dict]:
        """Generate explanations."""
        stage_start = time.time()
        
        if self.explainer:
            for c in candidates[:self.config.final_top_k]:
                explanation = self.explainer.explain(
                    inputs["ingredient"],
                    c["name"],
                    context=inputs["context"]
                )
                c["explanation"] = explanation.summary if explanation else ""
                c["tips"] = explanation.usage_tips if explanation else []
                c["warnings"] = explanation.warnings if explanation else []
                c["ratio"] = explanation.ratio if explanation else "1:1"
        
        self.stage_times["explain"] = time.time() - stage_start
        return candidates
    
    def _add_confidence(self, candidates: List[Dict], inputs: Dict) -> List[Dict]:
        """Add confidence estimates."""
        stage_start = time.time()
        
        if self.confidence_scorer:
            for c in candidates:
                c["confidence"] = self.confidence_scorer.compute_confidence(
                    np.array([c["score"]])
                )
        else:
            # Simple heuristic confidence
            for c in candidates:
                c["confidence"] = min(c["score"] * 1.1, 0.99)
        
        self.stage_times["confidence"] = time.time() - stage_start
        return candidates
    
    def _postprocess(self, candidates: List[Dict], inputs: Dict) -> List[Dict]:
        """Final postprocessing."""
        stage_start = time.time()
        
        # Ensure all expected fields
        for c in candidates:
            c.setdefault("explanation", "")
            c.setdefault("tips", [])
            c.setdefault("warnings", [])
            c.setdefault("ratio", "1:1")
            c.setdefault("confidence", 0.5)
        
        self.stage_times["postprocess"] = time.time() - stage_start
        return candidates
    
    def get_timing_report(self) -> Dict[str, float]:
        """Get timing breakdown by stage."""
        return self.stage_times.copy()
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        return {
            "retriever": self.retriever is not None,
            "dietary_checker": self.dietary_checker is not None,
            "flavor_matcher": self.flavor_matcher is not None,
            "texture_matcher": self.texture_matcher is not None,
            "reranker": self.reranker is not None,
            "explainer": self.explainer is not None,
            "confidence_scorer": self.confidence_scorer is not None,
        }


class PipelineEvaluator:
    """Evaluate pipeline performance."""
    
    def __init__(self, pipeline: SubstitutionPipeline):
        self.pipeline = pipeline
    
    def evaluate(
        self,
        test_cases: List[Dict],
        ground_truth: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate pipeline on test cases.
        
        Args:
            test_cases: List of test inputs
            ground_truth: Ingredient -> correct substitutes
            
        Returns:
            Evaluation metrics
        """
        # Your solution here
        pass
    
    def compute_recall_at_k(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """Compute recall@k."""
        # Your solution here
        pass
    
    def compute_mrr(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        # Your solution here
        pass
    
    def benchmark_latency(
        self,
        test_cases: List[Dict],
        num_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark pipeline latency."""
        # Your solution here
        pass


class PipelineSerializer:
    """Serialize and load pipelines."""
    
    @staticmethod
    def save(pipeline: SubstitutionPipeline, path: str):
        """Save pipeline to directory."""
        # Your solution here
        pass
    
    @staticmethod
    def load(path: str) -> SubstitutionPipeline:
        """Load pipeline from directory."""
        # Your solution here
        pass


def run_pipeline_demo():
    """Demo the pipeline with mock components."""
    print("=" * 60)
    print("Ingredient Substitution Pipeline Demo")
    print("=" * 60)
    
    # Create mock components
    class MockRetriever:
        def find_similar(self, ingredient, top_k=10):
            substitutes = {
                "butter": [("margarine", 0.92), ("coconut_oil", 0.85), 
                          ("olive_oil", 0.75), ("avocado", 0.65)],
                "egg": [("flax_egg", 0.88), ("applesauce", 0.75),
                       ("banana", 0.70), ("aquafaba", 0.80)],
                "milk": [("oat_milk", 0.90), ("almond_milk", 0.88),
                        ("soy_milk", 0.85), ("coconut_milk", 0.75)],
            }
            return substitutes.get(ingredient, [])[:top_k]
    
    class MockDietaryChecker:
        def check(self, ingredient, requirements):
            vegan = {"margarine", "coconut_oil", "olive_oil", "avocado",
                    "flax_egg", "applesauce", "banana", "aquafaba",
                    "oat_milk", "almond_milk", "soy_milk", "coconut_milk"}
            if "vegan" in requirements:
                return ingredient in vegan
            return True
        
        def is_safe(self, ingredient, allergens):
            allergen_map = {"almond_milk": ["tree_nuts"]}
            ing_allergens = allergen_map.get(ingredient, [])
            return not any(a in ing_allergens for a in allergens)
    
    # Create pipeline
    config = PipelineConfig(
        retrieval_top_k=20,
        final_top_k=3,
        use_reranking=False,
        generate_explanations=False
    )
    
    pipeline = SubstitutionPipeline(
        config=config,
        embedding_retriever=MockRetriever(),
        dietary_checker=MockDietaryChecker()
    )
    
    # Run demo queries
    print("\n1. Basic substitution:")
    result = pipeline.run("butter")
    print(result.summary())
    print(f"   Processing time: {result.processing_time_ms:.1f}ms")
    
    print("\n2. Vegan substitution:")
    result = pipeline.run(
        "butter",
        dietary_requirements=["vegan"]
    )
    print(result.summary())
    
    print("\n3. Recipe-aware substitution:")
    result = pipeline.run(
        "egg",
        recipe=["flour", "sugar", "butter", "chocolate"],
        dietary_requirements=["vegan"],
        context="chocolate brownies"
    )
    print(result.summary())
    
    print("\n4. With allergen avoidance:")
    result = pipeline.run(
        "milk",
        dietary_requirements=["vegan"],
        allergens_to_avoid=["tree_nuts"]
    )
    print(result.summary())
    
    print("\n5. Pipeline health check:")
    health = pipeline.health_check()
    for component, status in health.items():
        status_str = "✓" if status else "✗"
        print(f"   {status_str} {component}")
    
    print("\n6. Timing breakdown:")
    timing = pipeline.get_timing_report()
    for stage, ms in timing.items():
        print(f"   {stage}: {ms*1000:.2f}ms")
    
    return pipeline


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing PipelineConfig...")
    
    # Test 1: Default config
    config = PipelineConfig()
    assert config.retrieval_top_k == 50, "Test 1 failed"
    print("  ✓ Default config created")
    
    # Test 2: From dict
    config_dict = {"retrieval_top_k": 100, "final_top_k": 10}
    config = PipelineConfig.from_dict(config_dict)
    assert config.retrieval_top_k == 100, "Test 2 failed"
    print("  ✓ Config from dict works")
    
    # Test 3: To dict
    config_out = config.to_dict()
    assert "retrieval_top_k" in config_out, "Test 3 failed"
    print("  ✓ Config to dict works")
    
    print("\nTesting SubstitutionResult...")
    
    # Test 4: Create result
    result = SubstitutionResult(
        original="butter",
        substitutes=[
            {"name": "margarine", "score": 0.92},
            {"name": "coconut_oil", "score": 0.85}
        ],
        recipe_context=["flour", "sugar"],
        dietary_requirements=["vegan"],
        processing_time_ms=15.5
    )
    assert result.get_top_substitute()["name"] == "margarine", "Test 4 failed"
    print("  ✓ Result created")
    
    # Test 5: Summary
    summary = result.summary()
    assert "margarine" in summary, "Test 5 failed"
    print(f"  ✓ Summary:\n{summary}")
    
    # Test 6: To dict
    result_dict = result.to_dict()
    assert "original" in result_dict, "Test 6 failed"
    print("  ✓ Result to dict works")
    
    print("\nTesting SubstitutionPipeline...")
    
    # Run the demo which includes tests
    pipeline = run_pipeline_demo()
    
    print("\nTesting PipelineEvaluator...")
    
    evaluator = PipelineEvaluator(pipeline)
    
    # Test 7: Recall@k
    recall = evaluator.compute_recall_at_k(
        ["margarine", "coconut_oil", "olive_oil"],
        ["margarine", "avocado"],
        k=3
    )
    assert 0 <= recall <= 1, f"Test 7 failed: {recall}"
    print(f"  ✓ Recall@3: {recall:.2f}")
    
    # Test 8: MRR
    mrr = evaluator.compute_mrr(
        ["margarine", "coconut_oil"],
        ["margarine"]
    )
    assert mrr == 1.0, f"Test 8 failed: {mrr}"
    print(f"  ✓ MRR: {mrr:.2f}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
    
    print("\n" + "="*60)
    print("CONGRATULATIONS!")
    print("="*60)
    print("""
You have completed Part 8: Building a Substitution Model!

You've learned:
1.  Introduction to ingredient substitution concepts
2.  Building substitution datasets with positive/negative pairs
3.  Rule-based substitution as a baseline
4.  Embedding-based retrieval for similarity search
5.  Context-aware substitution considering recipe type
6.  Binary classification for substitution prediction
7.  Pairwise ranking for ordering substitutes
8.  Siamese networks for learning embeddings
9.  Cross-encoders for accurate scoring
10. Bi-encoders for efficient retrieval
11. Hard negative mining for better training
12. Dietary constraint handling for safety
13. Flavor profile matching for taste
14. Texture and function matching
15. Multi-ingredient substitution
16. Explanation generation for transparency
17. Confidence scoring and uncertainty
18. Building production APIs
19. End-to-end pipeline integration

Next Steps:
- Part 9: Evaluation and Iteration
- Fine-tune models on real substitution data
- Deploy the API in production
- Collect user feedback for improvement
""")
