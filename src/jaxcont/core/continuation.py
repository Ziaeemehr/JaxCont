"""
Core continuation problem definition and solution containers.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax.numpy as jnp
from jax import Array


def _json_safe(value):
    """Recursively convert JAX/NumPy arrays into plain Python lists so a
    value is safe to pass to ``json.dumps`` — used for the metadata blob
    :meth:`ContinuationSolution.save` embeds in its ``.npz`` archive."""
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


_OPTIONAL_ARRAYS = ("tangent_vectors", "eigenvalues", "stability", "trivial_index",
                    "floquet_logabs", "floquet_converged")


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
    rhs: Callable[[Array, dict[str, float]], Array]
    u0: Array
    params: dict[str, float]
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
        state_names: Optional display names for each state component
        param_name: Optional display name for the continuation parameter
        trivial_index: Periodic flow-direction multiplier index per point
        floquet_logabs: Periodic ``log|multiplier|`` per point
        floquet_converged: Periodic spectrum status per point
        curve_layout: ``CurveLayout`` of a two-parameter curve branch
        rejected_events: Failed refinements, shaped like ``bifurcations``
        status: Lower-case ``Status`` name of the run
    """
    states: Array
    parameters: Array
    eigenvalues: Optional[Array] = None
    stability: Optional[Array] = None
    bifurcations: list[dict[str, Any]] = field(default_factory=list)
    tangent_vectors: Optional[Array] = None
    convergence_info: Optional[list[dict[str, Any]]] = None
    state_names: Optional[tuple[str, ...]] = None
    param_name: Optional[str] = None
    trivial_index: Optional[Array] = None
    floquet_logabs: Optional[Array] = None
    floquet_converged: Optional[Array] = None
    curve_layout: Any = None
    rejected_events: list[dict[str, Any]] = field(default_factory=list)
    status: Optional[str] = None
    
    @property
    def n_points(self) -> int:
        """Number of solution points."""
        return len(self.parameters)
    
    @property
    def state_dim(self) -> int:
        """Dimension of state space."""
        return self.states.shape[1] if len(self.states.shape) > 1 else 1
    
    def get_point(self, index: int) -> tuple[Array, float]:
        """
        Get state and parameter at a specific point.
        
        Args:
            index: Point index
        
        Returns:
            (state, parameter_value) tuple
        """
        return self.states[index], float(self.parameters[index])
    
    def get_bifurcations_by_type(self, bif_type: str) -> list[dict[str, Any]]:
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
        from jaxcont.viz import plot_continuation
        return plot_continuation(self, **kwargs)
    
    def save(self, filename: str):
        """
        Save this solution to ``filename`` (an ``.npz`` archive), using a
        versioned, pickle-free schema (format_version=2; :meth:`load` also
        reads version 1, which lacks the fields added in version 2).

        Numeric arrays (states, parameters, tangent_vectors, eigenvalues,
        stability, trivial_index, floquet_logabs, floquet_converged) are
        stored as native NumPy arrays; an optional array
        that is ``None`` is simply omitted from the archive rather than
        stored as ``None`` -- NumPy has no null array value, and storing
        ``None`` via ``np.array(None)`` both requires pickling on load and
        round-trips as a 0-d object array, not ``None``. Non-numeric
        metadata (bifurcations, rejected_events, convergence_info,
        state_names, param_name, status and the curve_layout fields)
        is JSON-encoded into a single ``metadata_json`` entry, which NumPy
        stores as a plain unicode array -- no pickling needed there either.
        Complex metadata values (including event multipliers) are encoded as
        ``[real, imaginary]`` pairs. Loading preserves these pairs; callers
        can recover a complex scalar with ``complex(*pair)``.

        Args:
            filename: Path to save file
        """
        import json

        import numpy as np

        arrays: dict[str, Any] = {"format_version": np.array(2)}
        arrays["states"] = np.asarray(self.states)
        arrays["parameters"] = np.asarray(self.parameters)
        for name in _OPTIONAL_ARRAYS:
            value = getattr(self, name)
            if value is not None:
                arrays[name] = np.asarray(value)

        layout = self.curve_layout
        metadata = {
            "bifurcations": _json_safe(self.bifurcations),
            "rejected_events": _json_safe(self.rejected_events),
            "convergence_info": _json_safe(self.convergence_info),
            "state_names": list(self.state_names) if self.state_names is not None else None,
            "param_name": self.param_name,
            "status": self.status,
            "curve_layout": None if layout is None else {
                "kind": layout.kind, "formulation": layout.formulation,
                "n": int(layout.n), "free": int(layout.free)},
        }
        arrays["metadata_json"] = np.array(json.dumps(metadata))

        np.savez(filename, **arrays)

    @classmethod
    def load(cls, filename: str) -> "ContinuationSolution":
        """
        Load a solution saved by :meth:`save`.

        Reads with ``allow_pickle=False``: the format_version 1 and 2
        schemas never need pickling, so this refuses to execute arbitrary pickled
        payloads from an untrusted ``.npz`` file rather than trusting them
        by default.

        Args:
            filename: Path to load file

        Returns:
            ContinuationSolution object
        """
        import json

        import numpy as np

        with np.load(filename, allow_pickle=False) as data:
            if "format_version" not in data.files:
                raise ValueError(
                    f"{filename!r} has no 'format_version' entry -- it is not "
                    f"a ContinuationSolution.save() archive (or predates this "
                    f"schema) and cannot be loaded."
                )
            version = int(data["format_version"])
            if version not in (1, 2):
                raise ValueError(
                    f"{filename!r} uses save/load format_version={version}, "
                    f"but this version of JaxCont only supports versions 1 and 2."
                )

            metadata = json.loads(data["metadata_json"].item())
            state_names = metadata["state_names"]
            layout = metadata.get("curve_layout")
            if layout is not None:
                from jaxcont.bifurcations.curve_layout import CurveLayout
                layout = CurveLayout(**layout)

            return cls(
                states=jnp.array(data["states"]),
                parameters=jnp.array(data["parameters"]),
                bifurcations=metadata["bifurcations"],
                convergence_info=metadata["convergence_info"],
                state_names=tuple(state_names) if state_names is not None else None,
                param_name=metadata["param_name"],
                curve_layout=layout,
                rejected_events=metadata.get("rejected_events", []),
                status=metadata.get("status"),
                **{name: jnp.array(data[name]) if name in data.files else None
                   for name in _OPTIONAL_ARRAYS},
            )
