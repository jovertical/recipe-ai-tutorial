"""
Exercise 19: Monitoring Training

In this exercise, you'll implement training monitoring utilities to track
progress, detect problems early, and make informed decisions about when
to stop training.

Example:
    logger = TrainingLogger()
    
    for step in range(1000):
        loss = train_step()
        logger.log({"loss": loss, "step": step})
        
        if logger.should_stop_early(patience=5):
            print("Early stopping triggered!")
            break
    
    logger.plot_metrics()

ML Relevance:
    Monitoring is essential for successful training:
    - Detect divergence (loss going up) early
    - Know when to stop (diminishing returns)
    - Diagnose problems (learning rate too high/low)
    - Compare experiments (which config is better)
    
    Without monitoring, you're training blind.

Your Task:
    1. Implement TrainingLogger class - log and retrieve metrics
    2. Implement detect_divergence() - catch training problems
    3. Implement should_stop_early() - early stopping logic
    4. Implement calculate_running_stats() - smoothed metrics
    5. Implement compare_runs() - compare multiple training runs

Run:
    python tutorials/06-fine-tuning/19_monitoring_training.py
"""

from typing import Dict, Any, List, Optional, Tuple
import math
from collections import defaultdict


class TrainingLogger:
    """
    Logger for tracking training metrics.
    
    Example:
        logger = TrainingLogger()
        logger.log({"loss": 2.5, "step": 0})
        logger.log({"loss": 2.3, "step": 1})
        
        history = logger.get_history("loss")
        # [2.5, 2.3]
    """
    
    def __init__(self, log_dir: str = None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Optional directory to save logs
        """
        # Your solution here
        # Hints:
        # 1. Create a dictionary to store metric histories
        # 2. Store log_dir for potential file logging
        # 3. Track current step
        pass
    
    def log(self, metrics: Dict[str, float], step: int = None) -> None:
        """
        Log metrics for a step.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (auto-incremented if None)
            
        Example:
            logger.log({"loss": 2.5, "lr": 0.001})
            logger.log({"loss": 2.3, "lr": 0.001}, step=10)
        """
        # Your solution here
        pass
    
    def get_history(self, metric_name: str) -> List[float]:
        """
        Get the full history of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of values for that metric
        """
        # Your solution here
        pass
    
    def get_last(self, metric_name: str, n: int = 1) -> List[float]:
        """
        Get the last n values of a metric.
        
        Args:
            metric_name: Name of the metric
            n: Number of values to retrieve
            
        Returns:
            List of last n values
        """
        # Your solution here
        pass
    
    def get_best(self, metric_name: str, mode: str = "min") -> Tuple[int, float]:
        """
        Get the best value and step for a metric.
        
        Args:
            metric_name: Name of the metric
            mode: "min" for lower is better, "max" for higher is better
            
        Returns:
            Tuple of (step, value) for best metric
        """
        # Your solution here
        pass
    
    def get_current_step(self) -> int:
        """Return the current step number."""
        # Your solution here
        pass


def calculate_running_stats(
    values: List[float],
    window_size: int = 10
) -> Dict[str, List[float]]:
    """
    Calculate running mean and standard deviation.
    
    Args:
        values: List of metric values
        window_size: Size of the rolling window
        
    Returns:
        Dictionary with "mean" and "std" lists
        
    Example:
        >>> values = [10, 12, 8, 11, 9, 13, 7, 14, 8, 12]
        >>> stats = calculate_running_stats(values, window_size=3)
        >>> len(stats["mean"])
        10  # Same length as input
    """
    # Your solution here
    # Hints:
    # 1. For positions with insufficient history, use available values
    # 2. Calculate mean and std for each window
    pass


def detect_divergence(
    loss_history: List[float],
    window_size: int = 10,
    threshold: float = 0.5
) -> Tuple[bool, str]:
    """
    Detect if training is diverging.
    
    Divergence indicators:
    - Loss is NaN or Inf
    - Loss is increasing over time
    - Loss has extremely high variance
    
    Args:
        loss_history: List of loss values
        window_size: Window for trend calculation
        threshold: Threshold for declaring divergence
        
    Returns:
        Tuple of (is_diverging, reason)
        
    Example:
        >>> losses = [2.5, 2.4, 2.3, 5.0, 10.0, float('nan')]
        >>> is_div, reason = detect_divergence(losses)
        >>> is_div
        True
        >>> "nan" in reason.lower()
        True
    """
    # Your solution here
    # Hints:
    # 1. Check for NaN/Inf first
    # 2. Compare recent mean to earlier mean
    # 3. Check if recent values have high variance
    pass


def should_stop_early(
    loss_history: List[float],
    patience: int = 5,
    min_delta: float = 0.001
) -> Tuple[bool, int]:
    """
    Check if training should stop early.
    
    Early stopping triggers when validation loss hasn't improved
    for 'patience' evaluation steps.
    
    Args:
        loss_history: List of loss values (typically validation loss)
        patience: Number of steps to wait for improvement
        min_delta: Minimum change to qualify as improvement
        
    Returns:
        Tuple of (should_stop, steps_without_improvement)
        
    Example:
        >>> losses = [2.0, 1.8, 1.7, 1.7, 1.7, 1.7, 1.7]
        >>> stop, wait = should_stop_early(losses, patience=5)
        >>> stop
        True
        >>> wait
        5
    """
    # Your solution here
    # Hints:
    # 1. Find the best loss so far
    # 2. Count steps since best loss
    # 3. Improvement requires min_delta decrease
    pass


def calculate_learning_rate_at_step(
    step: int,
    total_steps: int,
    base_lr: float = 1e-4,
    warmup_steps: int = 100,
    scheduler_type: str = "cosine"
) -> float:
    """
    Calculate learning rate at a given step.
    
    Args:
        step: Current step
        total_steps: Total training steps
        base_lr: Base/maximum learning rate
        warmup_steps: Number of warmup steps
        scheduler_type: "linear", "cosine", or "constant"
        
    Returns:
        Learning rate at this step
        
    Example:
        >>> calculate_learning_rate_at_step(0, 1000, base_lr=1e-4, warmup_steps=100)
        0.0  # Start of warmup
        >>> calculate_learning_rate_at_step(100, 1000, base_lr=1e-4, warmup_steps=100)
        0.0001  # End of warmup, at base_lr
    """
    # Your solution here
    # Hints:
    # 1. During warmup: linear increase from 0 to base_lr
    # 2. After warmup: apply scheduler_type decay
    # 3. For cosine: use cosine annealing formula
    pass


def estimate_remaining_time(
    current_step: int,
    total_steps: int,
    elapsed_seconds: float
) -> Dict[str, float]:
    """
    Estimate remaining training time.
    
    Args:
        current_step: Current training step
        total_steps: Total steps to complete
        elapsed_seconds: Time elapsed so far
        
    Returns:
        Dictionary with time estimates
        
    Example:
        >>> result = estimate_remaining_time(100, 1000, 60.0)
        >>> result["seconds_per_step"]
        0.6
        >>> result["remaining_seconds"]
        540.0  # 900 remaining steps * 0.6
    """
    # Your solution here
    pass


def compare_runs(
    runs: Dict[str, List[float]],
    metric_name: str = "loss"
) -> Dict[str, Any]:
    """
    Compare multiple training runs.
    
    Args:
        runs: Dictionary of run_name -> loss history
        metric_name: Name of metric being compared
        
    Returns:
        Dictionary with comparison results
        
    Example:
        >>> runs = {
        ...     "run_a": [2.5, 2.0, 1.5, 1.2],
        ...     "run_b": [2.5, 2.2, 2.0, 1.8]
        ... }
        >>> result = compare_runs(runs)
        >>> result["best_run"]
        'run_a'
        >>> result["final_values"]["run_a"]
        1.2
    """
    # Your solution here
    # Hints:
    # 1. Find final value for each run
    # 2. Find best final value
    # 3. Calculate improvement rate for each
    pass


def summarize_training(
    logger: TrainingLogger,
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    Create a summary of the training run.
    
    Args:
        logger: TrainingLogger with recorded metrics
        metrics: List of metrics to summarize (default: all)
        
    Returns:
        Dictionary with training summary
        
    Example:
        >>> summary = summarize_training(logger)
        >>> summary["total_steps"]
        1000
        >>> summary["metrics"]["loss"]["final"]
        0.5
        >>> summary["metrics"]["loss"]["best"]
        0.45
    """
    # Your solution here
    pass


class EarlyStopping:
    """
    Early stopping handler for training loops.
    
    Example:
        early_stop = EarlyStopping(patience=5, min_delta=0.001)
        
        for epoch in range(100):
            val_loss = evaluate()
            if early_stop(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Steps to wait for improvement
            min_delta: Minimum improvement to reset patience
            mode: "min" for loss, "max" for accuracy
        """
        # Your solution here
        pass
    
    def __call__(self, metric: float) -> bool:
        """
        Check if should stop.
        
        Args:
            metric: Current metric value
            
        Returns:
            True if should stop, False otherwise
        """
        # Your solution here
        pass
    
    def reset(self) -> None:
        """Reset the early stopping state."""
        # Your solution here
        pass
    
    @property
    def best_value(self) -> float:
        """Return the best metric value seen."""
        # Your solution here
        pass
    
    @property
    def steps_without_improvement(self) -> int:
        """Return steps since last improvement."""
        # Your solution here
        pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    print("Testing TrainingLogger...")
    
    # Test 1: Basic logging
    logger = TrainingLogger()
    logger.log({"loss": 2.5, "lr": 0.001})
    logger.log({"loss": 2.3, "lr": 0.001})
    logger.log({"loss": 2.1, "lr": 0.001})
    
    history = logger.get_history("loss")
    assert history == [2.5, 2.3, 2.1], f"Test 1a failed: {history}"
    assert logger.get_current_step() == 3, f"Test 1b failed"
    print("  ✓ Basic logging works")
    
    # Test 2: Get last n
    last = logger.get_last("loss", n=2)
    assert last == [2.3, 2.1], f"Test 2 failed: {last}"
    print("  ✓ Get last n works")
    
    # Test 3: Get best
    step, value = logger.get_best("loss", mode="min")
    assert value == 2.1, f"Test 3a failed: {value}"
    assert step == 2, f"Test 3b failed: {step}"
    print("  ✓ Get best works")
    
    print("\nTesting calculate_running_stats...")
    
    # Test 4: Running stats
    values = [10.0, 12.0, 8.0, 11.0, 9.0]
    stats = calculate_running_stats(values, window_size=3)
    assert len(stats["mean"]) == 5, f"Test 4a failed"
    assert len(stats["std"]) == 5, f"Test 4b failed"
    # Last 3 values: [11, 9] wait that's 2... let me fix
    # Actually window=3 means [8, 11, 9] for last position
    # Mean should be (8+11+9)/3 = 9.33
    assert abs(stats["mean"][-1] - (8+11+9)/3) < 0.01, f"Test 4c failed: {stats['mean'][-1]}"
    print("  ✓ Running stats calculated")
    
    print("\nTesting detect_divergence...")
    
    # Test 5: Detect NaN
    losses = [2.5, 2.4, 2.3, float('nan')]
    is_div, reason = detect_divergence(losses)
    assert is_div, "Test 5a failed"
    assert "nan" in reason.lower(), f"Test 5b failed: {reason}"
    print("  ✓ Detects NaN")
    
    # Test 6: Detect increasing loss
    losses = [2.0, 2.1, 2.3, 2.5, 2.8, 3.2, 3.8, 4.5, 5.0, 6.0]
    is_div, reason = detect_divergence(losses, window_size=5, threshold=0.3)
    assert is_div, f"Test 6 failed: should detect divergence, reason: {reason}"
    print("  ✓ Detects increasing loss")
    
    # Test 7: Normal training should not trigger
    losses = [2.5, 2.3, 2.1, 2.0, 1.9, 1.85, 1.82, 1.80, 1.78, 1.76]
    is_div, _ = detect_divergence(losses, window_size=5, threshold=0.3)
    assert not is_div, "Test 7 failed: normal training flagged as divergent"
    print("  ✓ Normal training not flagged")
    
    print("\nTesting should_stop_early...")
    
    # Test 8: Should stop
    losses = [2.0, 1.8, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
    stop, wait = should_stop_early(losses, patience=5, min_delta=0.01)
    assert stop, f"Test 8a failed: should stop, wait={wait}"
    assert wait >= 5, f"Test 8b failed: wait={wait}"
    print("  ✓ Early stopping triggers correctly")
    
    # Test 9: Should not stop
    losses = [2.0, 1.8, 1.6, 1.4, 1.2]
    stop, wait = should_stop_early(losses, patience=5, min_delta=0.01)
    assert not stop, f"Test 9 failed: should not stop"
    print("  ✓ Continues when improving")
    
    print("\nTesting calculate_learning_rate_at_step...")
    
    # Test 10: Warmup start
    lr = calculate_learning_rate_at_step(0, 1000, base_lr=1e-4, warmup_steps=100)
    assert lr == 0.0, f"Test 10 failed: {lr}"
    print("  ✓ LR starts at 0")
    
    # Test 11: End of warmup
    lr = calculate_learning_rate_at_step(100, 1000, base_lr=1e-4, warmup_steps=100)
    assert abs(lr - 1e-4) < 1e-8, f"Test 11 failed: {lr}"
    print("  ✓ LR reaches base at warmup end")
    
    # Test 12: Mid warmup
    lr = calculate_learning_rate_at_step(50, 1000, base_lr=1e-4, warmup_steps=100)
    assert abs(lr - 5e-5) < 1e-8, f"Test 12 failed: {lr}"
    print("  ✓ Linear warmup correct")
    
    # Test 13: Cosine decay
    lr = calculate_learning_rate_at_step(
        1000, 1000, base_lr=1e-4, warmup_steps=100, scheduler_type="cosine"
    )
    assert lr < 1e-5, f"Test 13 failed: lr should be near 0 at end, got {lr}"
    print("  ✓ Cosine decay works")
    
    print("\nTesting estimate_remaining_time...")
    
    # Test 14: Time estimation
    result = estimate_remaining_time(100, 1000, 60.0)
    assert abs(result["seconds_per_step"] - 0.6) < 0.01, f"Test 14a failed"
    assert abs(result["remaining_seconds"] - 540.0) < 1, f"Test 14b failed"
    print("  ✓ Time estimation correct")
    
    print("\nTesting compare_runs...")
    
    # Test 15: Run comparison
    runs = {
        "run_a": [2.5, 2.0, 1.5, 1.2],
        "run_b": [2.5, 2.2, 2.0, 1.8]
    }
    result = compare_runs(runs)
    assert result["best_run"] == "run_a", f"Test 15a failed: {result['best_run']}"
    assert result["final_values"]["run_a"] == 1.2, f"Test 15b failed"
    assert result["final_values"]["run_b"] == 1.8, f"Test 15c failed"
    print("  ✓ Run comparison works")
    
    print("\nTesting EarlyStopping class...")
    
    # Test 16: Early stopping object
    early_stop = EarlyStopping(patience=3, min_delta=0.01)
    
    assert not early_stop(2.0), "Test 16a failed"  # First value, best so far
    assert not early_stop(1.8), "Test 16b failed"  # Improved
    assert not early_stop(1.7), "Test 16c failed"  # Improved
    assert not early_stop(1.7), "Test 16d failed"  # No improvement, wait=1
    assert not early_stop(1.69), "Test 16e failed"  # Barely improved
    assert not early_stop(1.69), "Test 16f failed"  # No improvement, wait=1
    assert not early_stop(1.69), "Test 16g failed"  # No improvement, wait=2
    assert not early_stop(1.69), "Test 16h failed"  # No improvement, wait=3
    assert early_stop(1.69), "Test 16i failed"  # Should stop now
    print("  ✓ EarlyStopping class works")
    
    # Test 17: Best value tracking
    early_stop = EarlyStopping(patience=5, mode="min")
    early_stop(2.0)
    early_stop(1.5)
    early_stop(1.8)
    assert early_stop.best_value == 1.5, f"Test 17 failed: {early_stop.best_value}"
    print("  ✓ Best value tracked")
    
    # Test 18: Reset
    early_stop.reset()
    assert early_stop.steps_without_improvement == 0, "Test 18a failed"
    print("  ✓ Reset works")
    
    print("\nTesting summarize_training...")
    
    # Test 19: Training summary
    logger = TrainingLogger()
    for i in range(10):
        logger.log({"loss": 2.0 - i * 0.1, "accuracy": 0.5 + i * 0.05})
    
    summary = summarize_training(logger)
    assert "total_steps" in summary, "Test 19a failed"
    assert summary["total_steps"] == 10, f"Test 19b failed: {summary['total_steps']}"
    assert "metrics" in summary, "Test 19c failed"
    assert "loss" in summary["metrics"], "Test 19d failed"
    print("  ✓ Training summary generated")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    print("\nYou've implemented training monitoring utilities.")
    print("Key takeaways:")
    print("- Track metrics to understand training progress")
    print("- Early stopping prevents overfitting")
    print("- Divergence detection catches problems early")
    print("- Good logging enables meaningful experiments")
