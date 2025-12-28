# Problem 19: Hyperparameter Tuning
#
# Implement hyperparameter optimization strategies.
#
# Example:
#   tuner = BayesianOptimizer(objective_fn, param_space)
#   best_params = tuner.optimize(n_trials=50)
#   # Returns: {"lr": 0.003, "batch_size": 64, "layers": 3}
#
# ML Relevance: Hyperparameter tuning:
# - Can significantly improve model performance
# - Systematic exploration beats random search
# - Enables fair model comparison

from typing import List, Dict, Callable, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class Trial:
    """Single hyperparameter trial."""
    trial_id: int
    params: Dict[str, Any]
    score: float
    duration: float = None


@dataclass
class ParamSpace:
    """Parameter search space."""
    name: str
    param_type: str  # "float", "int", "categorical"
    low: float = None
    high: float = None
    choices: List[Any] = None
    log_scale: bool = False


def random_search(
    objective_fn: Callable,
    param_space: List[ParamSpace],
    n_trials: int = 20
) -> Tuple[Dict[str, Any], float]:
    """
    Random hyperparameter search.

    Args:
        objective_fn: Function that takes params and returns score
        param_space: Search space definition
        n_trials: Number of random trials

    Returns:
        (best_params, best_score)
    """
    # Your solution here
    pass


def grid_search(
    objective_fn: Callable,
    param_grid: Dict[str, List[Any]]
) -> Tuple[Dict[str, Any], float]:
    """
    Exhaustive grid search.

    Args:
        objective_fn: Objective function
        param_grid: {param_name: [values to try]}

    Returns:
        (best_params, best_score)
    """
    # Your solution here
    pass


class BayesianOptimizer:
    """Bayesian optimization for hyperparameters."""

    def __init__(
        self,
        objective_fn: Callable,
        param_space: List[ParamSpace],
        n_initial: int = 5
    ):
        """
        Initialize optimizer.

        Args:
            objective_fn: Function to maximize
            param_space: Search space
            n_initial: Initial random samples before modeling
        """
        # Your solution here
        pass

    def sample_params(self) -> Dict[str, Any]:
        """Sample next parameters to try."""
        # Your solution here
        pass

    def update(self, params: Dict[str, Any], score: float):
        """Update model with new observation."""
        # Your solution here
        pass

    def optimize(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Run optimization.

        Returns:
            Best parameters found
        """
        # Your solution here
        pass

    def get_trials(self) -> List[Trial]:
        """Get all trials."""
        # Your solution here
        pass


class SuccessiveHalving:
    """Successive halving for efficient tuning."""

    def __init__(
        self,
        objective_fn: Callable,
        param_space: List[ParamSpace],
        max_budget: int = 100,
        reduction_factor: int = 3
    ):
        """Initialize successive halving."""
        # Your solution here
        pass

    def optimize(self, n_configs: int = 27) -> Dict[str, Any]:
        """
        Run successive halving.

        Starts with many configs, low budget.
        Keeps best fraction, increases budget.
        """
        # Your solution here
        pass


def learning_rate_finder(
    train_fn: Callable,
    lr_range: Tuple[float, float] = (1e-7, 1.0),
    n_steps: int = 100
) -> float:
    """
    Find optimal learning rate by training with increasing LR.

    Returns:
        Suggested learning rate
    """
    # Your solution here
    pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    # Simple quadratic objective for testing
    def objective(params):
        # Optimal at lr=0.01, layers=3
        lr_score = -((params["lr"] - 0.01) ** 2) * 1000
        layer_score = -((params["layers"] - 3) ** 2)
        return lr_score + layer_score + np.random.normal(0, 0.1)

    param_space = [
        ParamSpace("lr", "float", low=0.0001, high=0.1, log_scale=True),
        ParamSpace("layers", "int", low=1, high=5),
    ]

    print("Testing hyperparameter tuning...")

    # Test 1: Random search
    best_params, best_score = random_search(objective, param_space, n_trials=20)
    assert "lr" in best_params and "layers" in best_params, "Test 1 failed"
    print(f"  ✓ Random search: lr={best_params['lr']:.4f}, layers={best_params['layers']}")

    # Test 2: Grid search
    param_grid = {"lr": [0.001, 0.01, 0.1], "layers": [2, 3, 4]}
    best_params, best_score = grid_search(objective, param_grid)
    assert best_params["lr"] == 0.01 or best_params["layers"] == 3, "Test 2 failed"
    print(f"  ✓ Grid search: {best_params}")

    # Test 3: Bayesian optimization
    optimizer = BayesianOptimizer(objective, param_space, n_initial=5)
    best_params = optimizer.optimize(n_trials=20)
    assert "lr" in best_params, "Test 3 failed"
    print(f"  ✓ Bayesian opt: lr={best_params['lr']:.4f}, layers={best_params['layers']}")

    # Test 4: Get trials
    trials = optimizer.get_trials()
    assert len(trials) == 20, f"Test 4 failed: {len(trials)}"
    print(f"  ✓ Tracked {len(trials)} trials")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
