"""
Exercise 03: Similarity Metrics

In this exercise, you'll implement different ways to measure similarity
between embedding vectors. Understanding these metrics is crucial for
evaluating and using embeddings effectively.

Example:
    vec1 = [1, 0, 0]
    vec2 = [1, 0, 0]  # Same direction
    vec3 = [0, 1, 0]  # Perpendicular
    vec4 = [-1, 0, 0] # Opposite direction
    
    cosine_similarity(vec1, vec2) = 1.0   # Identical
    cosine_similarity(vec1, vec3) = 0.0   # Unrelated
    cosine_similarity(vec1, vec4) = -1.0  # Opposite

ML Relevance:
    Different similarity metrics have different properties:
    - Cosine: Measures angle, ignores magnitude (most common for embeddings)
    - Euclidean: Measures distance, sensitive to magnitude
    - Dot product: Fast, used in attention mechanisms
    
    Choosing the right metric affects retrieval and classification quality.

Your Task:
    1. Implement cosine_similarity() - angle between vectors
    2. Implement euclidean_distance() - straight-line distance
    3. Implement dot_product() - unnormalized similarity  
    4. Implement batch versions for efficiency
    5. Compare metrics on real examples

Run:
    python tutorials/07-embeddings/03_similarity_metrics.py
"""

from typing import List, Dict, Tuple
import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    cosine_similarity = dot(a, b) / (||a|| * ||b||)
    
    Args:
        vec1: First vector (1D array)
        vec2: Second vector (1D array)
        
    Returns:
        Cosine similarity between -1 and 1
        
    Example:
        >>> cosine_similarity(np.array([1, 0]), np.array([1, 0]))
        1.0
        >>> cosine_similarity(np.array([1, 0]), np.array([0, 1]))
        0.0
        >>> cosine_similarity(np.array([1, 0]), np.array([-1, 0]))
        -1.0
    """
    # Your solution here
    # Hints:
    # 1. Compute dot product
    # 2. Compute norms
    # 3. Handle zero vectors (return 0)
    pass


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    distance = sqrt(sum((a - b)^2))
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance (>= 0)
        
    Example:
        >>> euclidean_distance(np.array([0, 0]), np.array([3, 4]))
        5.0
        >>> euclidean_distance(np.array([1, 1]), np.array([1, 1]))
        0.0
    """
    # Your solution here
    pass


def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate dot product between two vectors.
    
    dot_product = sum(a * b)
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Dot product (can be any real number)
        
    Example:
        >>> dot_product(np.array([1, 2, 3]), np.array([4, 5, 6]))
        32  # 1*4 + 2*5 + 3*6
    """
    # Your solution here
    pass


def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Manhattan (L1) distance between two vectors.
    
    distance = sum(|a - b|)
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Manhattan distance (>= 0)
        
    Example:
        >>> manhattan_distance(np.array([0, 0]), np.array([3, 4]))
        7.0  # |3| + |4|
    """
    # Your solution here
    pass


def pairwise_cosine_similarity(
    matrix1: np.ndarray,
    matrix2: np.ndarray = None
) -> np.ndarray:
    """
    Calculate pairwise cosine similarities between all vector pairs.
    
    Args:
        matrix1: First set of vectors (n, d)
        matrix2: Second set of vectors (m, d), or None to compare matrix1 to itself
        
    Returns:
        Similarity matrix (n, m) or (n, n) if matrix2 is None
        
    Example:
        >>> vecs = np.array([[1, 0], [0, 1], [1, 1]])
        >>> sims = pairwise_cosine_similarity(vecs)
        >>> sims.shape
        (3, 3)
        >>> sims[0, 0]  # Self-similarity
        1.0
    """
    # Your solution here
    # Hints:
    # 1. Normalize each vector
    # 2. Use matrix multiplication for all dot products at once
    pass


def pairwise_euclidean_distance(
    matrix1: np.ndarray,
    matrix2: np.ndarray = None
) -> np.ndarray:
    """
    Calculate pairwise Euclidean distances between all vector pairs.
    
    Args:
        matrix1: First set of vectors (n, d)
        matrix2: Second set of vectors (m, d), or None for self-comparison
        
    Returns:
        Distance matrix (n, m) or (n, n)
        
    Example:
        >>> vecs = np.array([[0, 0], [3, 4]])
        >>> dists = pairwise_euclidean_distance(vecs)
        >>> dists[0, 1]
        5.0
    """
    # Your solution here
    # Hints:
    # Use the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
    pass


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors (make them unit length).
    
    Args:
        vectors: Array of vectors (n, d)
        
    Returns:
        Normalized vectors where each has L2 norm of 1
        
    Example:
        >>> vecs = np.array([[3, 4], [0, 5]])
        >>> normed = normalize_vectors(vecs)
        >>> np.linalg.norm(normed[0])  
        1.0
    """
    # Your solution here
    # Handle zero vectors (keep them as zeros)
    pass


def similarity_to_distance(
    similarity: float,
    method: str = "angular"
) -> float:
    """
    Convert similarity score to distance.
    
    Args:
        similarity: Cosine similarity (-1 to 1)
        method: Conversion method
            "angular" -> arccos(similarity) / pi (0 to 1)
            "inverse" -> 1 - similarity (0 to 2)
            
    Returns:
        Distance value
        
    Example:
        >>> similarity_to_distance(1.0, "angular")
        0.0
        >>> similarity_to_distance(0.0, "angular")
        0.5
        >>> similarity_to_distance(-1.0, "angular")
        1.0
    """
    # Your solution here
    pass


def find_most_similar(
    query: np.ndarray,
    candidates: np.ndarray,
    metric: str = "cosine",
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Find most similar vectors to a query.
    
    Args:
        query: Query vector (d,)
        candidates: Candidate vectors (n, d)
        metric: Similarity metric ("cosine", "euclidean", "dot")
        top_k: Number of results to return
        
    Returns:
        List of (index, score) tuples, sorted by similarity
        For "euclidean", lower distance = more similar
        
    Example:
        >>> query = np.array([1, 0])
        >>> candidates = np.array([[1, 0], [0, 1], [0.7, 0.7]])
        >>> results = find_most_similar(query, candidates, "cosine", top_k=2)
        >>> results[0]
        (0, 1.0)  # Exact match
    """
    # Your solution here
    pass


def compare_metrics(
    word_pairs: List[Tuple[str, str]],
    embeddings: Dict[str, np.ndarray]
) -> Dict[str, List[float]]:
    """
    Compare different similarity metrics on word pairs.
    
    Args:
        word_pairs: List of (word1, word2) tuples
        embeddings: Word to embedding dictionary
        
    Returns:
        Dictionary with metric name -> list of scores
        
    Example:
        >>> pairs = [("king", "queen"), ("king", "banana")]
        >>> result = compare_metrics(pairs, embeddings)
        >>> result["cosine"]
        [0.85, 0.12]  # High for similar, low for different
    """
    # Your solution here
    pass


def analyze_metric_properties(
    vectors: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Analyze properties of different metrics on a set of vectors.
    
    Args:
        vectors: Set of vectors (n, d)
        
    Returns:
        Dictionary with metric analysis:
        - mean, std, min, max for each metric
        
    Example:
        >>> vecs = np.random.randn(100, 64)
        >>> analysis = analyze_metric_properties(vecs)
        >>> "cosine" in analysis
        True
    """
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    print("Testing cosine_similarity...")
    
    # Test 1: Identical vectors
    v1 = np.array([1.0, 0.0, 0.0])
    sim = cosine_similarity(v1, v1)
    assert abs(sim - 1.0) < 1e-6, f"Test 1 failed: {sim}"
    print("  ✓ Identical vectors = 1.0")
    
    # Test 2: Perpendicular vectors
    v2 = np.array([0.0, 1.0, 0.0])
    sim = cosine_similarity(v1, v2)
    assert abs(sim) < 1e-6, f"Test 2 failed: {sim}"
    print("  ✓ Perpendicular vectors = 0.0")
    
    # Test 3: Opposite vectors
    v3 = np.array([-1.0, 0.0, 0.0])
    sim = cosine_similarity(v1, v3)
    assert abs(sim + 1.0) < 1e-6, f"Test 3 failed: {sim}"
    print("  ✓ Opposite vectors = -1.0")
    
    # Test 4: Scale invariance
    v4 = np.array([2.0, 0.0, 0.0])  # Same direction, different magnitude
    sim = cosine_similarity(v1, v4)
    assert abs(sim - 1.0) < 1e-6, f"Test 4 failed: {sim}"
    print("  ✓ Cosine is scale-invariant")
    
    print("\nTesting euclidean_distance...")
    
    # Test 5: Same point
    dist = euclidean_distance(v1, v1)
    assert abs(dist) < 1e-6, f"Test 5 failed: {dist}"
    print("  ✓ Same point distance = 0.0")
    
    # Test 6: Known distance
    p1 = np.array([0.0, 0.0])
    p2 = np.array([3.0, 4.0])
    dist = euclidean_distance(p1, p2)
    assert abs(dist - 5.0) < 1e-6, f"Test 6 failed: {dist}"
    print("  ✓ 3-4-5 triangle distance = 5.0")
    
    print("\nTesting dot_product...")
    
    # Test 7: Basic dot product
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    dp = dot_product(v1, v2)
    assert dp == 32, f"Test 7 failed: {dp}"
    print("  ✓ Dot product calculation correct")
    
    print("\nTesting manhattan_distance...")
    
    # Test 8: Manhattan distance
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    dist = manhattan_distance(p1, p2)
    assert dist == 7.0, f"Test 8 failed: {dist}"
    print("  ✓ Manhattan distance correct")
    
    print("\nTesting pairwise_cosine_similarity...")
    
    # Test 9: Self-comparison
    vecs = np.array([[1, 0], [0, 1], [1, 1]])
    sims = pairwise_cosine_similarity(vecs)
    assert sims.shape == (3, 3), f"Test 9a failed: {sims.shape}"
    assert abs(sims[0, 0] - 1.0) < 1e-6, f"Test 9b failed: diagonal should be 1"
    assert abs(sims[0, 1]) < 1e-6, f"Test 9c failed: perpendicular should be 0"
    print("  ✓ Pairwise cosine works")
    
    # Test 10: Cross comparison
    vecs1 = np.array([[1, 0], [0, 1]])
    vecs2 = np.array([[1, 0], [1, 1], [-1, 0]])
    sims = pairwise_cosine_similarity(vecs1, vecs2)
    assert sims.shape == (2, 3), f"Test 10 failed: {sims.shape}"
    print("  ✓ Cross-set comparison works")
    
    print("\nTesting pairwise_euclidean_distance...")
    
    # Test 11: Pairwise distance
    vecs = np.array([[0, 0], [3, 4], [6, 8]])
    dists = pairwise_euclidean_distance(vecs)
    assert dists.shape == (3, 3), f"Test 11a failed: {dists.shape}"
    assert abs(dists[0, 0]) < 1e-6, f"Test 11b failed: self-distance should be 0"
    assert abs(dists[0, 1] - 5.0) < 1e-6, f"Test 11c failed: {dists[0, 1]}"
    print("  ✓ Pairwise euclidean works")
    
    print("\nTesting normalize_vectors...")
    
    # Test 12: Normalization
    vecs = np.array([[3, 4], [0, 5], [0, 0]])
    normed = normalize_vectors(vecs)
    assert abs(np.linalg.norm(normed[0]) - 1.0) < 1e-6, "Test 12a failed"
    assert abs(np.linalg.norm(normed[1]) - 1.0) < 1e-6, "Test 12b failed"
    assert np.allclose(normed[2], 0), "Test 12c failed: zero vector should stay zero"
    print("  ✓ Normalization works")
    
    print("\nTesting similarity_to_distance...")
    
    # Test 13: Angular distance
    dist = similarity_to_distance(1.0, "angular")
    assert abs(dist) < 1e-6, f"Test 13a failed: {dist}"
    dist = similarity_to_distance(0.0, "angular")
    assert abs(dist - 0.5) < 1e-6, f"Test 13b failed: {dist}"
    dist = similarity_to_distance(-1.0, "angular")
    assert abs(dist - 1.0) < 1e-6, f"Test 13c failed: {dist}"
    print("  ✓ Angular distance conversion works")
    
    # Test 14: Inverse distance
    dist = similarity_to_distance(1.0, "inverse")
    assert abs(dist) < 1e-6, f"Test 14a failed: {dist}"
    dist = similarity_to_distance(0.0, "inverse")
    assert abs(dist - 1.0) < 1e-6, f"Test 14b failed: {dist}"
    print("  ✓ Inverse distance conversion works")
    
    print("\nTesting find_most_similar...")
    
    # Test 15: Find similar with cosine
    query = np.array([1.0, 0.0])
    candidates = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    results = find_most_similar(query, candidates, "cosine", top_k=2)
    assert len(results) == 2, f"Test 15a failed: {len(results)}"
    assert results[0][0] == 0, f"Test 15b failed: {results[0]}"
    assert abs(results[0][1] - 1.0) < 1e-6, f"Test 15c failed: {results[0][1]}"
    print("  ✓ Find most similar works")
    
    # Test 16: Find similar with euclidean
    results = find_most_similar(query, candidates, "euclidean", top_k=2)
    assert results[0][0] == 0, f"Test 16 failed: {results[0]}"  # Exact match has distance 0
    print("  ✓ Euclidean search works")
    
    print("\nTesting compare_metrics...")
    
    # Test 17: Compare metrics
    embeddings = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.9, 0.1]),
        "c": np.array([0.0, 1.0])
    }
    pairs = [("a", "b"), ("a", "c")]
    result = compare_metrics(pairs, embeddings)
    assert "cosine" in result, "Test 17a failed"
    assert "euclidean" in result, "Test 17b failed"
    assert len(result["cosine"]) == 2, "Test 17c failed"
    print("  ✓ Metric comparison works")
    
    print("\nTesting analyze_metric_properties...")
    
    # Test 18: Analyze metrics
    np.random.seed(42)
    vectors = np.random.randn(50, 32)
    analysis = analyze_metric_properties(vectors)
    assert "cosine" in analysis, "Test 18a failed"
    assert "mean" in analysis["cosine"], "Test 18b failed"
    assert "std" in analysis["cosine"], "Test 18c failed"
    print("  ✓ Metric analysis works")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    print("\nKey takeaways:")
    print("- Cosine similarity measures angle (scale-invariant)")
    print("- Euclidean distance measures straight-line distance")
    print("- Dot product is fast but not normalized")
    print("- Choose metric based on your use case")
