"""
Core continuation problem definition and solution containers.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Tuple, List
import jax.numpy as jnp
from jax import Array


@dataclass
class ContinuationProblem:
    """
    Definition of a continuation problem.
    
    Attributes:
        rhs: Right-hand side function f(u, params) defining the dynamical system
        u0: Initial state vector
        params: Dictionary of system parameters
        continuation_param: Name of the parameter to continue
        constraints: Optional additional constraints
        problem_type: Type of problem ('equilibrium', 'periodic', 'bvp')
    """
    rhs: Callable[[Array, Dict[str, float]], Array]
    u0: Array
    params: Dict[str, float]
    continuation_param: str
    constraints: Optional[Callable] = None
    problem_type: str = "equilibrium"
    
    def __post_init__(self):
        """Validate the problem definition."""
        if self.continuation_param not in self.params:
            raise ValueError(
                f"Continuation parameter '{self.continuation_param}' "
                f"not found in params: {list(self.params.keys())}"
            )
        
        # Validate problem type
        valid_types = ["equilibrium", "periodic", "bvp", "homoclinic"]
        if self.problem_type not in valid_types:
            raise ValueError(
                f"Problem type '{self.problem_type}' not recognized. "
                f"Valid types: {valid_types}"
            )
    
    @property
    def state_dim(self) -> int:
        """Dimension of the state space."""
        return len(self.u0)
    
    @property
    def param_value(self) -> float:
        """Current value of the continuation parameter."""
        return self.params[self.continuation_param]
    
    def evaluate_rhs(self, u: Array, param_value: Optional[float] = None) -> Array:
        """
        Evaluate the right-hand side at given state and parameter value.
        
        Args:
            u: State vector
            param_value: Value of continuation parameter (uses current if None)
        
        Returns:
            Value of f(u, params)
        """
        params = self.params.copy()
        if param_value is not None:
            params[self.continuation_param] = param_value
        return self.rhs(u, params)


@dataclass
class ContinuationSolution:
    """
    Container for continuation solution data.
    
    Attributes:
        states: Array of solution states along the branch (n_points, state_dim)
        parameters: Array of parameter values (n_points,)
        eigenvalues: Optional array of eigenvalues at each point
        stability: Array of stability indicators
        bifurcations: List of detected bifurcation points
        tangent_vectors: Tangent vectors along the branch
        convergence_info: Information about convergence at each step
    """
    states: Array
    parameters: Array
    eigenvalues: Optional[Array] = None
    stability: Optional[Array] = None
    bifurcations: List[Dict[str, Any]] = field(default_factory=list)
    tangent_vectors: Optional[Array] = None
    convergence_info: Optional[List[Dict[str, Any]]] = None
    
    @property
    def n_points(self) -> int:
        """Number of solution points."""
        return len(self.parameters)
    
    @property
    def state_dim(self) -> int:
        """Dimension of state space."""
        return self.states.shape[1] if len(self.states.shape) > 1 else 1
    
    def get_point(self, index: int) -> Tuple[Array, float]:
        """
        Get state and parameter at a specific point.
        
        Args:
            index: Point index
        
        Returns:
            (state, parameter_value) tuple
        """
        return self.states[index], float(self.parameters[index])
    
    def get_bifurcations_by_type(self, bif_type: str) -> List[Dict[str, Any]]:
        """
        Get all bifurcations of a specific type.
        
        Args:
            bif_type: Type of bifurcation ('fold', 'hopf', 'period-doubling', etc.)
        
        Returns:
            List of bifurcation dictionaries
        """
        return [bif for bif in self.bifurcations if bif.get("type") == bif_type]
    
    def is_stable(self, index: int) -> bool:
        """
        Check if solution is stable at given index.
        
        Args:
            index: Point index
        
        Returns:
            True if stable, False otherwise
        """
        if self.stability is None:
            raise ValueError("Stability information not available")
        return bool(self.stability[index])
    
    def plot(self, **kwargs):
        """
        Plot the continuation diagram.
        
        Args:
            **kwargs: Additional plotting options
        """
        from jaxcont.utils.plotting import plot_continuation
        return plot_continuation(self, **kwargs)
    
    def save(self, filename: str):
        """
        Save solution to file.
        
        Args:
            filename: Path to save file
        """
        import numpy as np
        data = {
            "states": np.array(self.states),
            "parameters": np.array(self.parameters),
            "eigenvalues": np.array(self.eigenvalues) if self.eigenvalues is not None else None,
            "stability": np.array(self.stability) if self.stability is not None else None,
            "bifurcations": self.bifurcations,
        }
        np.savez(filename, **data)
    
    @classmethod
    def load(cls, filename: str) -> "ContinuationSolution":
        """
        Load solution from file.
        
        Args:
            filename: Path to load file
        
        Returns:
            ContinuationSolution object
        """
        import numpy as np
        data = np.load(filename, allow_pickle=True)
        return cls(
            states=jnp.array(data["states"]),
            parameters=jnp.array(data["parameters"]),
            eigenvalues=jnp.array(data["eigenvalues"]) if data["eigenvalues"] is not None else None,
            stability=jnp.array(data["stability"]) if data["stability"] is not None else None,
            bifurcations=list(data["bifurcations"]) if "bifurcations" in data else [],
        )


def equilibrium_continuation(
    problem: ContinuationProblem,
    param_range: Tuple[float, float],
    **kwargs
) -> ContinuationSolution:
    """
    Perform equilibrium continuation.
    
    Args:
        problem: Continuation problem definition
        param_range: (min, max) parameter range for continuation
        **kwargs: Additional options (ds, max_steps, etc.)
    
    Returns:
        ContinuationSolution object
    """
    from jaxcont.core.pseudo_arclength import PseudoArclengthContinuation
    
    if problem.problem_type != "equilibrium":
        raise ValueError(f"Expected equilibrium problem, got {problem.problem_type}")
    
    # Create continuation method
    continuation = PseudoArclengthContinuation(**kwargs)
    
    # Run continuation
    solution = continuation.run(problem, param_range)
    
    return solution


def periodic_continuation(
    problem: ContinuationProblem,
    param_range: Tuple[float, float],
    **kwargs
) -> ContinuationSolution:
    """
    Perform periodic orbit continuation.
    
    Args:
        problem: Continuation problem definition
        param_range: (min, max) parameter range for continuation
        **kwargs: Additional options (ds, max_steps, method, etc.)
    
    Returns:
        ContinuationSolution object
    """
    from jaxcont.core.pseudo_arclength import PseudoArclengthContinuation
    
    if problem.problem_type != "periodic":
        raise ValueError(f"Expected periodic problem, got {problem.problem_type}")
    
    # Create continuation method
    continuation = PseudoArclengthContinuation(**kwargs)
    
    # Run continuation
    solution = continuation.run(problem, param_range)
    
    return solution
