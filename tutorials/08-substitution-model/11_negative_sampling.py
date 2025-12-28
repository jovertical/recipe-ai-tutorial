# Problem 11: Hard Negative Mining
#
# Implement hard negative mining to improve model training by focusing on
# difficult examples that the model currently gets wrong.
#
# Example:
#   miner = HardNegativeMiner(model, margin=0.2)
#   hard_negatives = miner.mine(anchor="butter", positives=["margarine"])
#   # Returns ingredients close to butter but NOT valid substitutes
#
# ML Relevance: Hard negatives are crucial for contrastive learning - they
# force the model to learn fine-grained distinctions rather than easy ones.
#
# Your Task:
#   1. Implement static hard negative mining
#   2. Implement online hard negative mining during training
#   3. Implement semi-hard negative selection
#   4. Compare training with random vs hard negatives


from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import random
import heapq


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


@dataclass
class NegativeSample:
    """A negative sample with metadata."""
    name: str
    embedding: np.ndarray
    similarity: float  # Similarity to anchor
    category: str = None
    difficulty: str = None  # "easy", "medium", "hard"
    
    def __repr__(self):
        return f"{self.name} (sim={self.similarity:.3f}, {self.difficulty})"


class NegativeSampler:
    """
    Base class for negative sampling strategies.
    """
    
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        positive_pairs: Dict[str, Set[str]],
        categories: Dict[str, str] = None
    ):
        """
        Initialize sampler.
        
        Args:
            embeddings: Ingredient embeddings
            positive_pairs: Ingredient -> set of valid substitutes
            categories: Ingredient -> category mapping
        """
        self.embeddings = embeddings
        self.positive_pairs = positive_pairs
        self.categories = categories or {}
        self.all_ingredients = list(embeddings.keys())
    
    def sample(
        self,
        anchor: str,
        num_samples: int = 5
    ) -> List[NegativeSample]:
        """
        Sample negatives for an anchor.
        
        Args:
            anchor: Anchor ingredient
            num_samples: Number of negatives to sample
            
        Returns:
            List of NegativeSample objects
        """
        raise NotImplementedError


class RandomNegativeSampler(NegativeSampler):
    """
    Random negative sampling.
    
    Simple baseline that randomly selects non-substitutes.
    """
    
    def sample(
        self,
        anchor: str,
        num_samples: int = 5
    ) -> List[NegativeSample]:
        """Sample random negatives."""
        # Your solution here
        # Hints:
        # 1. Get positive pairs for anchor
        # 2. Exclude anchor and positives
        # 3. Randomly sample from remaining
        pass


class HardNegativeSampler(NegativeSampler):
    """
    Hard negative sampling based on embedding similarity.
    
    Selects negatives that are close in embedding space but not substitutes.
    """
    
    def sample(
        self,
        anchor: str,
        num_samples: int = 5,
        similarity_threshold: float = 0.9
    ) -> List[NegativeSample]:
        """
        Sample hard negatives.
        
        Args:
            anchor: Anchor ingredient
            num_samples: Number of negatives
            similarity_threshold: Max similarity to include (avoid false negatives)
            
        Returns:
            Hard negative samples
        """
        # Your solution here
        # Hints:
        # 1. Compute similarity to all non-positives
        # 2. Filter out very high similarity (might be unlabeled positives)
        # 3. Select highest similarity negatives
        pass


class SemiHardNegativeSampler(NegativeSampler):
    """
    Semi-hard negative sampling.
    
    Selects negatives that are harder than the hardest positive but not too hard.
    This is used in triplet loss training.
    """
    
    def sample_for_triplet(
        self,
        anchor: str,
        positive: str,
        num_samples: int = 1
    ) -> List[NegativeSample]:
        """
        Sample semi-hard negatives for a triplet.
        
        Semi-hard: d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin
        
        Args:
            anchor: Anchor ingredient
            positive: Positive ingredient
            num_samples: Number of negatives
            
        Returns:
            Semi-hard negatives
        """
        # Your solution here
        pass
    
    def sample(
        self,
        anchor: str,
        num_samples: int = 5
    ) -> List[NegativeSample]:
        """Sample semi-hard negatives using average positive distance."""
        # Your solution here
        pass


class CategoryAwareNegativeSampler(NegativeSampler):
    """
    Category-aware negative sampling.
    
    Samples more negatives from the same category (harder) and fewer from different
    categories (easier) for a balanced difficulty curriculum.
    """
    
    def sample(
        self,
        anchor: str,
        num_samples: int = 5,
        same_category_ratio: float = 0.6
    ) -> List[NegativeSample]:
        """
        Sample with category awareness.
        
        Args:
            anchor: Anchor ingredient
            num_samples: Total negatives
            same_category_ratio: Fraction from same category
            
        Returns:
            Category-balanced negatives
        """
        # Your solution here
        pass
    
    def get_same_category_negatives(
        self,
        anchor: str,
        num_samples: int
    ) -> List[NegativeSample]:
        """Get negatives from same category as anchor."""
        # Your solution here
        pass


class MixedNegativeSampler(NegativeSampler):
    """
    Mixed negative sampling combining multiple strategies.
    """
    
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        positive_pairs: Dict[str, Set[str]],
        categories: Dict[str, str] = None,
        easy_ratio: float = 0.3,
        medium_ratio: float = 0.4,
        hard_ratio: float = 0.3
    ):
        """
        Initialize mixed sampler.
        
        Args:
            easy_ratio: Fraction of easy negatives
            medium_ratio: Fraction of medium negatives
            hard_ratio: Fraction of hard negatives
        """
        super().__init__(embeddings, positive_pairs, categories)
        self.easy_ratio = easy_ratio
        self.medium_ratio = medium_ratio
        self.hard_ratio = hard_ratio
    
    def sample(
        self,
        anchor: str,
        num_samples: int = 5
    ) -> List[NegativeSample]:
        """Sample mix of easy, medium, and hard negatives."""
        # Your solution here
        pass


class OnlineHardNegativeMiner:
    """
    Online hard negative mining during training.
    
    Mines hard negatives from the current batch instead of pre-computing.
    """
    
    def __init__(self, margin: float = 0.2):
        """
        Initialize miner.
        
        Args:
            margin: Margin for semi-hard definition
        """
        self.margin = margin
    
    def mine_batch(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Mine hard triplets from a batch.
        
        Args:
            embeddings: Batch embeddings (n x dim)
            labels: Batch labels (n,)
            
        Returns:
            List of (anchor_idx, positive_idx, negative_idx)
        """
        # Your solution here
        pass
    
    def mine_hardest_triplets(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        max_triplets: int = None
    ) -> List[Tuple[int, int, int]]:
        """
        Mine only the hardest triplets.
        
        For each anchor, find hardest positive and hardest negative.
        """
        # Your solution here
        pass
    
    def mine_semihard_triplets(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Mine semi-hard triplets.
        
        Negatives that violate margin but are still harder than positives.
        """
        # Your solution here
        pass


class NegativeCache:
    """
    Cache for storing and reusing hard negatives.
    
    Improves efficiency by reusing previously mined hard negatives.
    """
    
    def __init__(self, cache_size: int = 1000, refresh_rate: float = 0.1):
        """
        Initialize cache.
        
        Args:
            cache_size: Maximum negatives to cache per anchor
            refresh_rate: Fraction of cache to refresh each epoch
        """
        self.cache_size = cache_size
        self.refresh_rate = refresh_rate
        self.cache: Dict[str, List[NegativeSample]] = defaultdict(list)
    
    def get(self, anchor: str, num_samples: int) -> List[NegativeSample]:
        """Get cached negatives for anchor."""
        # Your solution here
        pass
    
    def update(
        self,
        anchor: str,
        new_negatives: List[NegativeSample],
        model_embeddings: Dict[str, np.ndarray] = None
    ):
        """
        Update cache with new negatives.
        
        Args:
            anchor: Anchor ingredient
            new_negatives: Newly mined negatives
            model_embeddings: Current model embeddings for re-scoring
        """
        # Your solution here
        pass
    
    def refresh(
        self,
        sampler: NegativeSampler,
        model_embeddings: Dict[str, np.ndarray]
    ):
        """
        Refresh cache with current model.
        
        Args:
            sampler: Sampler to use for new negatives
            model_embeddings: Current model embeddings
        """
        # Your solution here
        pass


class CurriculumNegativeSampler:
    """
    Curriculum learning for negative sampling.
    
    Starts with easy negatives and progressively increases difficulty.
    """
    
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        positive_pairs: Dict[str, Set[str]],
        categories: Dict[str, str] = None,
        num_epochs: int = 100
    ):
        """
        Initialize curriculum sampler.
        
        Args:
            num_epochs: Total training epochs for scheduling
        """
        self.embeddings = embeddings
        self.positive_pairs = positive_pairs
        self.categories = categories
        self.num_epochs = num_epochs
        self.current_epoch = 0
    
    def get_difficulty_schedule(self, epoch: int) -> Dict[str, float]:
        """
        Get difficulty distribution for current epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with ratios for easy/medium/hard
        """
        # Your solution here
        # Example schedule:
        # Early: 70% easy, 20% medium, 10% hard
        # Middle: 30% easy, 40% medium, 30% hard
        # Late: 10% easy, 30% medium, 60% hard
        pass
    
    def sample(
        self,
        anchor: str,
        num_samples: int = 5,
        epoch: int = None
    ) -> List[NegativeSample]:
        """Sample negatives based on curriculum schedule."""
        # Your solution here
        pass
    
    def set_epoch(self, epoch: int):
        """Update current epoch for scheduling."""
        self.current_epoch = epoch


def classify_negative_difficulty(
    anchor_emb: np.ndarray,
    negative_emb: np.ndarray,
    positive_embs: List[np.ndarray]
) -> str:
    """
    Classify difficulty of a negative sample.
    
    Args:
        anchor_emb: Anchor embedding
        negative_emb: Negative embedding
        positive_embs: List of positive embeddings
        
    Returns:
        Difficulty level: "easy", "medium", or "hard"
    """
    # Your solution here
    pass


def analyze_negatives(
    samples: List[NegativeSample],
    anchor: str,
    positive_pairs: Dict[str, Set[str]]
) -> Dict[str, any]:
    """
    Analyze negative sample distribution.
    
    Args:
        samples: Negative samples
        anchor: Anchor ingredient
        positive_pairs: Ground truth positives
        
    Returns:
        Analysis dictionary
    """
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    embedding_dim = 64
    
    # Create mock embeddings
    categories = {
        "butter": "fat", "margarine": "fat", "coconut_oil": "fat", 
        "lard": "fat", "shortening": "fat", "olive_oil": "fat",
        "milk": "dairy", "cream": "dairy", "yogurt": "dairy",
        "oat_milk": "milk_alt", "almond_milk": "milk_alt",
        "sugar": "sweetener", "honey": "sweetener",
        "salt": "seasoning", "pepper": "seasoning",
        "flour": "grain", "rice": "grain"
    }
    
    # Create embeddings where same-category items are similar
    embeddings = {}
    category_vectors = {}
    for cat in set(categories.values()):
        category_vectors[cat] = np.random.randn(embedding_dim).astype(np.float32)
        category_vectors[cat] /= np.linalg.norm(category_vectors[cat])
    
    for ing, cat in categories.items():
        emb = category_vectors[cat] + np.random.randn(embedding_dim) * 0.2
        embeddings[ing] = (emb / np.linalg.norm(emb)).astype(np.float32)
    
    # Define positive pairs (valid substitutions)
    positive_pairs = {
        "butter": {"margarine", "coconut_oil"},
        "margarine": {"butter", "coconut_oil"},
        "milk": {"oat_milk", "almond_milk"},
        "sugar": {"honey"},
    }
    
    print("Testing RandomNegativeSampler...")
    
    # Test 1: Random sampling
    random_sampler = RandomNegativeSampler(embeddings, positive_pairs, categories)
    samples = random_sampler.sample("butter", num_samples=5)
    assert len(samples) == 5, f"Test 1a failed: {len(samples)}"
    for s in samples:
        assert s.name not in positive_pairs.get("butter", set()), f"Test 1b failed: {s.name}"
        assert s.name != "butter", f"Test 1c failed: sampled anchor"
    print(f"  ✓ Random samples: {[s.name for s in samples]}")
    
    print("\nTesting HardNegativeSampler...")
    
    # Test 2: Hard negative sampling
    hard_sampler = HardNegativeSampler(embeddings, positive_pairs, categories)
    samples = hard_sampler.sample("butter", num_samples=5)
    assert len(samples) <= 5, f"Test 2a failed: {len(samples)}"
    # Hard negatives should be from same category (fats)
    fat_count = sum(1 for s in samples if categories.get(s.name) == "fat")
    print(f"  ✓ Hard samples: {[s.name for s in samples]}")
    print(f"    Same category (fat): {fat_count}/{len(samples)}")
    
    # Test 3: Similarity ordering
    if len(samples) > 1:
        for i in range(len(samples) - 1):
            assert samples[i].similarity >= samples[i+1].similarity, "Test 3 failed: not sorted"
    print("  ✓ Sorted by similarity (hardest first)")
    
    print("\nTesting SemiHardNegativeSampler...")
    
    # Test 4: Semi-hard sampling
    semihard_sampler = SemiHardNegativeSampler(embeddings, positive_pairs, categories)
    samples = semihard_sampler.sample_for_triplet("butter", "margarine", num_samples=3)
    print(f"  ✓ Semi-hard samples for (butter, margarine): {[s.name for s in samples]}")
    
    print("\nTesting CategoryAwareNegativeSampler...")
    
    # Test 5: Category-aware sampling
    category_sampler = CategoryAwareNegativeSampler(embeddings, positive_pairs, categories)
    samples = category_sampler.sample("butter", num_samples=10, same_category_ratio=0.6)
    same_cat = sum(1 for s in samples if categories.get(s.name) == "fat")
    diff_cat = len(samples) - same_cat
    print(f"  ✓ Category-aware: {same_cat} same category, {diff_cat} different")
    
    print("\nTesting MixedNegativeSampler...")
    
    # Test 6: Mixed sampling
    mixed_sampler = MixedNegativeSampler(
        embeddings, positive_pairs, categories,
        easy_ratio=0.3, medium_ratio=0.4, hard_ratio=0.3
    )
    samples = mixed_sampler.sample("butter", num_samples=10)
    difficulty_counts = defaultdict(int)
    for s in samples:
        difficulty_counts[s.difficulty] += 1
    print(f"  ✓ Mixed samples: {dict(difficulty_counts)}")
    
    print("\nTesting OnlineHardNegativeMiner...")
    
    # Test 7: Online mining
    miner = OnlineHardNegativeMiner(margin=0.2)
    batch_embs = np.array([embeddings[ing] for ing in list(embeddings.keys())[:8]])
    batch_labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])  # 3 classes
    
    triplets = miner.mine_batch(batch_embs, batch_labels)
    assert len(triplets) > 0, "Test 7 failed"
    for a, p, n in triplets:
        assert batch_labels[a] == batch_labels[p], "Test 7b: anchor-positive mismatch"
        assert batch_labels[a] != batch_labels[n], "Test 7c: anchor-negative match"
    print(f"  ✓ Mined {len(triplets)} triplets from batch")
    
    # Test 8: Hardest triplets
    hardest = miner.mine_hardest_triplets(batch_embs, batch_labels, max_triplets=5)
    assert len(hardest) <= 5, f"Test 8 failed: {len(hardest)}"
    print(f"  ✓ Hardest triplets: {len(hardest)}")
    
    print("\nTesting NegativeCache...")
    
    # Test 9: Cache operations
    cache = NegativeCache(cache_size=100, refresh_rate=0.1)
    samples = hard_sampler.sample("butter", num_samples=10)
    cache.update("butter", samples)
    
    cached = cache.get("butter", 5)
    assert len(cached) <= 5, f"Test 9 failed: {len(cached)}"
    print(f"  ✓ Cached and retrieved {len(cached)} samples")
    
    print("\nTesting CurriculumNegativeSampler...")
    
    # Test 10: Curriculum scheduling
    curriculum = CurriculumNegativeSampler(
        embeddings, positive_pairs, categories,
        num_epochs=100
    )
    
    # Early epoch - should be mostly easy
    schedule_early = curriculum.get_difficulty_schedule(5)
    assert schedule_early["easy"] > schedule_early["hard"], "Test 10a failed"
    
    # Late epoch - should be mostly hard
    schedule_late = curriculum.get_difficulty_schedule(95)
    assert schedule_late["hard"] > schedule_late["easy"], "Test 10b failed"
    
    print(f"  ✓ Epoch 5 schedule: {schedule_early}")
    print(f"  ✓ Epoch 95 schedule: {schedule_late}")
    
    # Test 11: Curriculum sampling
    samples_early = curriculum.sample("butter", num_samples=10, epoch=5)
    samples_late = curriculum.sample("butter", num_samples=10, epoch=95)
    
    early_hard = sum(1 for s in samples_early if s.difficulty == "hard")
    late_hard = sum(1 for s in samples_late if s.difficulty == "hard")
    print(f"  ✓ Hard samples: early={early_hard}, late={late_hard}")
    
    print("\nTesting classify_negative_difficulty...")
    
    # Test 12: Difficulty classification
    anchor_emb = embeddings["butter"]
    positive_embs = [embeddings[p] for p in positive_pairs.get("butter", [])]
    
    easy_emb = embeddings["salt"]  # Different category
    hard_emb = embeddings["lard"]  # Same category
    
    easy_difficulty = classify_negative_difficulty(anchor_emb, easy_emb, positive_embs)
    hard_difficulty = classify_negative_difficulty(anchor_emb, hard_emb, positive_embs)
    
    print(f"  ✓ salt difficulty: {easy_difficulty}")
    print(f"  ✓ lard difficulty: {hard_difficulty}")
    
    print("\nTesting analyze_negatives...")
    
    # Test 13: Analysis
    samples = mixed_sampler.sample("butter", num_samples=20)
    analysis = analyze_negatives(samples, "butter", positive_pairs)
    assert "num_samples" in analysis, "Test 13a failed"
    assert "difficulty_distribution" in analysis, "Test 13b failed"
    print(f"  ✓ Analysis: {analysis}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Random negatives are simple but inefficient")
    print("2. Hard negatives improve discrimination")
    print("3. Semi-hard negatives balance difficulty")
    print("4. Online mining adapts to model state")
    print("5. Curriculum learning eases training")
    print("\nNext: Dietary constraint handling!")
