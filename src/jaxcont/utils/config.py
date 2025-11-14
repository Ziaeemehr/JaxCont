"""
Configuration management for JaxCont.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """
    Global configuration for JaxCont.
    
    Attributes:
        use_jit: Whether to use JAX JIT compilation
        use_gpu: Whether to use GPU acceleration
        default_continuation_method: Default continuation method
        default_ds: Default step size
        default_tolerance: Default tolerance for Newton solver
        plot_style: Default plotting style
    """
    use_jit: bool = True
    use_gpu: bool = False
    default_continuation_method: str = "pseudo-arclength"
    default_ds: float = 0.01
    default_ds_min: float = 1e-5
    default_ds_max: float = 0.1
    default_tolerance: float = 1e-6
    default_max_steps: int = 1000
    plot_style: str = "seaborn"
    
    # Bifurcation detection settings
    detect_fold: bool = True
    detect_hopf: bool = True
    detect_branch_point: bool = False
    
    # Stability analysis settings
    compute_stability: bool = True
    compute_eigenvalues: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.default_ds <= 0:
            raise ValueError("default_ds must be positive")
        if self.default_ds_min <= 0:
            raise ValueError("default_ds_min must be positive")
        if self.default_ds_max <= self.default_ds_min:
            raise ValueError("default_ds_max must be greater than default_ds_min")
        if self.default_tolerance <= 0:
            raise ValueError("default_tolerance must be positive")
    
    @classmethod
    def default(cls) -> "Config":
        """Get default configuration."""
        return cls()
    
    @classmethod
    def fast(cls) -> "Config":
        """Get configuration optimized for speed."""
        return cls(
            use_jit=True,
            default_ds=0.05,
            compute_eigenvalues=False,
        )
    
    @classmethod
    def accurate(cls) -> "Config":
        """Get configuration optimized for accuracy."""
        return cls(
            default_ds=0.001,
            default_tolerance=1e-8,
            compute_eigenvalues=True,
            compute_stability=True,
        )


# Global configuration instance
_global_config = Config.default()


def get_config() -> Config:
    """Get global configuration."""
    return _global_config


def set_config(config: Config):
    """Set global configuration."""
    global _global_config
    _global_config = config


def reset_config():
    """Reset to default configuration."""
    global _global_config
    _global_config = Config.default()
