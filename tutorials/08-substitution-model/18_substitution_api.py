# Problem 18: Substitution API
#
# Build a production-ready API for ingredient substitution.
#
# Example:
#   api = SubstitutionAPI(model, index)
#   response = api.get_substitutes("butter", context="baking", dietary=["vegan"])
#   # Returns: {"substitutes": [...], "confidence": 0.9, "explanations": [...]}
#
# ML Relevance: Production APIs require batching, caching, rate limiting,
# and graceful degradation - beyond just model inference.
#
# Your Task:
#   1. Implement request handling and validation
#   2. Implement response formatting
#   3. Implement caching and batching
#   4. Implement error handling and fallbacks


from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import time
import hashlib
from collections import OrderedDict
import numpy as np


class APIStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class SubstitutionRequest:
    """API request for substitution."""
    ingredient: str
    context: Optional[str] = None
    dietary: List[str] = field(default_factory=list)
    allergens_to_avoid: List[str] = field(default_factory=list)
    top_k: int = 5
    include_explanation: bool = True
    include_confidence: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate request parameters."""
        errors = []
        
        if not self.ingredient or not self.ingredient.strip():
            errors.append("ingredient is required")
        
        if self.top_k < 1 or self.top_k > 20:
            errors.append("top_k must be between 1 and 20")
        
        return len(errors) == 0, errors
    
    def to_cache_key(self) -> str:
        """Generate cache key for this request."""
        key_parts = [
            self.ingredient.lower(),
            self.context or "",
            ",".join(sorted(self.dietary)),
            ",".join(sorted(self.allergens_to_avoid)),
            str(self.top_k)
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


@dataclass
class SubstituteResult:
    """A single substitute in the response."""
    name: str
    score: float
    confidence: float = None
    ratio: str = None
    explanation: str = None
    warnings: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)


@dataclass
class SubstitutionResponse:
    """API response for substitution."""
    status: str
    ingredient: str
    substitutes: List[SubstituteResult]
    request_id: str = None
    processing_time_ms: float = None
    cached: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "ingredient": self.ingredient,
            "substitutes": [asdict(s) for s in self.substitutes],
            "request_id": self.request_id,
            "processing_time_ms": self.processing_time_ms,
            "cached": self.cached,
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class LRUCache:
    """Simple LRU cache for API responses."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries
        """
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            self.delete(key)
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove oldest
                oldest_key = next(iter(self.cache))
                self.delete(oldest_key)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def delete(self, key: str):
        """Delete from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.timestamps.clear()
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": 0  # Would track hits/misses in production
        }


class SubstitutionAPI:
    """
    Main API for ingredient substitution.
    """
    
    def __init__(
        self,
        retriever=None,  # Embedding retriever
        ranker=None,  # Ranking model
        explainer=None,  # Explanation generator
        constraint_checker=None,  # Dietary constraint checker
        cache_size: int = 1000,
        enable_caching: bool = True
    ):
        """
        Initialize API.
        
        Args:
            retriever: Embedding-based retriever
            ranker: Ranking/scoring model
            explainer: Explanation generator
            constraint_checker: Dietary constraint checker
            cache_size: LRU cache size
            enable_caching: Whether to enable caching
        """
        self.retriever = retriever
        self.ranker = ranker
        self.explainer = explainer
        self.constraint_checker = constraint_checker
        self.cache = LRUCache(cache_size) if enable_caching else None
        self.request_counter = 0
    
    def substitute(
        self,
        request: SubstitutionRequest
    ) -> SubstitutionResponse:
        """
        Process substitution request.
        
        Args:
            request: Substitution request
            
        Returns:
            Substitution response
        """
        start_time = time.time()
        self.request_counter += 1
        request_id = f"req_{self.request_counter}_{int(start_time)}"
        
        # Validate request
        is_valid, errors = request.validate()
        if not is_valid:
            return SubstitutionResponse(
                status=APIStatus.ERROR.value,
                ingredient=request.ingredient,
                substitutes=[],
                request_id=request_id,
                errors=errors
            )
        
        # Check cache
        if self.cache:
            cache_key = request.to_cache_key()
            cached = self.cache.get(cache_key)
            if cached:
                cached.request_id = request_id
                cached.cached = True
                cached.processing_time_ms = (time.time() - start_time) * 1000
                return cached
        
        # Process request
        try:
            substitutes = self._find_substitutes(request)
            
            response = SubstitutionResponse(
                status=APIStatus.SUCCESS.value,
                ingredient=request.ingredient,
                substitutes=substitutes,
                request_id=request_id,
                processing_time_ms=(time.time() - start_time) * 1000,
                cached=False
            )
            
            # Cache response
            if self.cache:
                self.cache.set(cache_key, response)
            
            return response
            
        except Exception as e:
            return SubstitutionResponse(
                status=APIStatus.ERROR.value,
                ingredient=request.ingredient,
                substitutes=[],
                request_id=request_id,
                processing_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
    
    def _find_substitutes(
        self,
        request: SubstitutionRequest
    ) -> List[SubstituteResult]:
        """Find substitutes for the request."""
        # Your solution here
        # 1. Use retriever to get candidates
        # 2. Filter by dietary constraints
        # 3. Rank candidates
        # 4. Generate explanations if requested
        # 5. Return top_k results
        pass
    
    def substitute_simple(
        self,
        ingredient: str,
        context: str = None,
        dietary: List[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Simple interface for substitution.
        
        Returns:
            List of substitute dictionaries
        """
        request = SubstitutionRequest(
            ingredient=ingredient,
            context=context,
            dietary=dietary or [],
            top_k=top_k
        )
        response = self.substitute(request)
        return [asdict(s) for s in response.substitutes]
    
    def batch_substitute(
        self,
        requests: List[SubstitutionRequest]
    ) -> List[SubstitutionResponse]:
        """
        Process multiple requests.
        
        Args:
            requests: List of requests
            
        Returns:
            List of responses
        """
        # Your solution here
        pass
    
    def health_check(self) -> Dict:
        """API health check."""
        return {
            "status": "healthy",
            "cache_stats": self.cache.stats() if self.cache else None,
            "requests_processed": self.request_counter
        }
    
    def get_stats(self) -> Dict:
        """Get API statistics."""
        return {
            "total_requests": self.request_counter,
            "cache": self.cache.stats() if self.cache else None
        }


class BatchProcessor:
    """Process substitution requests in batches for efficiency."""
    
    def __init__(
        self,
        api: SubstitutionAPI,
        batch_size: int = 32
    ):
        """
        Initialize batch processor.
        
        Args:
            api: SubstitutionAPI instance
            batch_size: Maximum batch size
        """
        self.api = api
        self.batch_size = batch_size
    
    def process(
        self,
        requests: List[SubstitutionRequest]
    ) -> List[SubstitutionResponse]:
        """Process requests in batches."""
        # Your solution here
        pass
    
    def process_ingredients(
        self,
        ingredients: List[str],
        **kwargs
    ) -> List[SubstitutionResponse]:
        """
        Convenience method for batch processing ingredients.
        
        Args:
            ingredients: List of ingredient names
            **kwargs: Additional parameters for all requests
        """
        requests = [
            SubstitutionRequest(ingredient=ing, **kwargs)
            for ing in ingredients
        ]
        return self.process(requests)


def create_api_from_config(config: Dict) -> SubstitutionAPI:
    """
    Create API from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured SubstitutionAPI
    """
    # Your solution here
    pass


def validate_ingredient(ingredient: str, known_ingredients: set) -> Tuple[bool, str]:
    """
    Validate ingredient name.
    
    Returns:
        Tuple of (is_valid, normalized_name or error)
    """
    # Your solution here
    pass


def format_response_for_display(response: SubstitutionResponse) -> str:
    """Format response for terminal display."""
    # Your solution here
    pass


# Mock implementations for testing
class MockRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, substitutes: Dict[str, List[Tuple[str, float]]]):
        self.substitutes = substitutes
    
    def find_similar(self, ingredient: str, top_k: int = 10):
        return self.substitutes.get(ingredient, [])[:top_k]


class MockConstraintChecker:
    """Mock constraint checker for testing."""
    
    def __init__(self, dietary_info: Dict[str, List[str]]):
        self.dietary_info = dietary_info
    
    def check(self, ingredient: str, requirements: List[str]) -> bool:
        tags = self.dietary_info.get(ingredient, [])
        return all(req in tags for req in requirements)


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing SubstitutionRequest...")
    
    # Test 1: Valid request
    request = SubstitutionRequest(
        ingredient="butter",
        context="baking",
        dietary=["vegan"],
        top_k=3
    )
    is_valid, errors = request.validate()
    assert is_valid, f"Test 1a failed: {errors}"
    print(f"  ✓ Valid request: {request.ingredient}")
    
    # Test 2: Invalid request
    bad_request = SubstitutionRequest(ingredient="", top_k=50)
    is_valid, errors = bad_request.validate()
    assert not is_valid, "Test 2 failed"
    assert len(errors) > 0, "Test 2b failed"
    print(f"  ✓ Invalid request errors: {errors}")
    
    # Test 3: Cache key
    key = request.to_cache_key()
    assert len(key) > 0, "Test 3 failed"
    print(f"  ✓ Cache key: {key}")
    
    print("\nTesting LRUCache...")
    
    cache = LRUCache(max_size=3, ttl_seconds=3600)
    
    # Test 4: Set and get
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1", "Test 4 failed"
    print("  ✓ Set and get works")
    
    # Test 5: LRU eviction
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    cache.set("key4", "value4")  # Should evict key1
    assert cache.get("key1") is None, "Test 5 failed"
    assert cache.get("key4") == "value4", "Test 5b failed"
    print("  ✓ LRU eviction works")
    
    # Test 6: Cache stats
    stats = cache.stats()
    assert "size" in stats, "Test 6 failed"
    print(f"  ✓ Cache stats: {stats}")
    
    print("\nTesting SubstitutionAPI...")
    
    # Create mock components
    mock_substitutes = {
        "butter": [
            ("margarine", 0.92),
            ("coconut_oil", 0.85),
            ("olive_oil", 0.75),
        ]
    }
    mock_dietary = {
        "margarine": ["vegan", "dairy_free"],
        "coconut_oil": ["vegan", "dairy_free"],
        "olive_oil": ["vegan", "dairy_free"],
        "butter": ["vegetarian"],
    }
    
    retriever = MockRetriever(mock_substitutes)
    checker = MockConstraintChecker(mock_dietary)
    
    api = SubstitutionAPI(
        retriever=retriever,
        constraint_checker=checker,
        cache_size=100,
        enable_caching=True
    )
    
    # Test 7: Basic substitution
    request = SubstitutionRequest(ingredient="butter", top_k=3)
    response = api.substitute(request)
    assert response.status == "success", f"Test 7 failed: {response.errors}"
    print(f"  ✓ Basic substitution: {len(response.substitutes)} results")
    
    # Test 8: Response fields
    assert response.request_id is not None, "Test 8a failed"
    assert response.processing_time_ms is not None, "Test 8b failed"
    print(f"  ✓ Request ID: {response.request_id}")
    print(f"  ✓ Processing time: {response.processing_time_ms:.2f}ms")
    
    # Test 9: Caching
    response2 = api.substitute(request)
    assert response2.cached == True, "Test 9 failed"
    print(f"  ✓ Second request was cached")
    
    # Test 10: Simple interface
    results = api.substitute_simple("butter", dietary=["vegan"])
    assert isinstance(results, list), "Test 10 failed"
    print(f"  ✓ Simple interface: {len(results)} results")
    
    # Test 11: Health check
    health = api.health_check()
    assert health["status"] == "healthy", "Test 11 failed"
    print(f"  ✓ Health check: {health['status']}")
    
    # Test 12: Stats
    stats = api.get_stats()
    assert "total_requests" in stats, "Test 12 failed"
    print(f"  ✓ API stats: {stats}")
    
    print("\nTesting SubstitutionResponse...")
    
    # Test 13: To dict
    response_dict = response.to_dict()
    assert "status" in response_dict, "Test 13 failed"
    print(f"  ✓ Response to dict: {list(response_dict.keys())}")
    
    # Test 14: To JSON
    response_json = response.to_json()
    parsed = json.loads(response_json)
    assert parsed["status"] == "success", "Test 14 failed"
    print(f"  ✓ Response to JSON: {len(response_json)} chars")
    
    print("\nTesting BatchProcessor...")
    
    processor = BatchProcessor(api, batch_size=10)
    
    # Test 15: Batch process
    requests = [
        SubstitutionRequest(ingredient="butter"),
        SubstitutionRequest(ingredient="butter", dietary=["vegan"]),
    ]
    responses = processor.process(requests)
    assert len(responses) == 2, f"Test 15 failed: {len(responses)}"
    print(f"  ✓ Batch processed: {len(responses)} requests")
    
    # Test 16: Process ingredients
    responses = processor.process_ingredients(
        ["butter"],
        context="baking"
    )
    assert len(responses) == 1, "Test 16 failed"
    print(f"  ✓ Process ingredients: {len(responses)} responses")
    
    print("\nTesting error handling...")
    
    # Test 17: Invalid ingredient
    bad_request = SubstitutionRequest(ingredient="")
    response = api.substitute(bad_request)
    assert response.status == "error", "Test 17 failed"
    assert len(response.errors) > 0, "Test 17b failed"
    print(f"  ✓ Error handling: {response.errors}")
    
    print("\nTesting format_response_for_display...")
    
    # Test 18: Display format
    good_request = SubstitutionRequest(ingredient="butter")
    response = api.substitute(good_request)
    display = format_response_for_display(response)
    assert len(display) > 0, "Test 18 failed"
    print(f"  ✓ Display format:\n{display}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Clean request/response interfaces")
    print("2. Validation prevents errors")
    print("3. Caching improves performance")
    print("4. Batch processing for efficiency")
    print("\nNext: End-to-end pipeline integration!")
