"""
Configuration management for JaxCont.
"""

import importlib
import pkgutil
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


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


def test_jax_cuda() -> Dict[str, any]:
    """
    Test JAX installation and CUDA/GPU support.
    
    Returns:
        Dictionary containing:
        - jax_available: bool
        - jax_version: str
        - devices: list of devices
        - default_backend: str
        - gpu_available: bool
        - cuda_available: bool
        - device_count: int
        - device_info: list of device details
    """
    result = {
        'jax_available': False,
        'jax_version': None,
        'devices': [],
        'default_backend': None,
        'gpu_available': False,
        'cuda_available': False,
        'device_count': 0,
        'device_info': [],
        'error': None
    }
    
    try:
        import jax
        import jax.numpy as jnp
        
        result['jax_available'] = True
        result['jax_version'] = jax.__version__
        
        # Get devices
        devices = jax.devices()
        result['devices'] = [str(d) for d in devices]
        result['device_count'] = len(devices)
        
        # Get default backend
        result['default_backend'] = jax.default_backend()
        
        # Check for GPU/CUDA
        result['gpu_available'] = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
        result['cuda_available'] = result['default_backend'] == 'gpu' or 'cuda' in result['default_backend'].lower()
        
        # Get detailed device info
        for i, device in enumerate(devices):
            device_dict = {
                'id': i,
                'device': str(device),
                'platform': device.platform,
                'device_kind': device.device_kind,
            }
            result['device_info'].append(device_dict)
        
        # Test a simple computation
        try:
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.sum(x)
            result['computation_test'] = 'passed'
        except Exception as e:
            result['computation_test'] = f'failed: {str(e)}'
            
    except ImportError as e:
        result['error'] = f'JAX import failed: {str(e)}'
    except Exception as e:
        result['error'] = f'Unexpected error: {str(e)}'
    
    return result


def print_jax_cuda_info():
    """Print formatted information about JAX and CUDA support."""
    info = test_jax_cuda()
    
    print("=" * 60)
    print("JAX and CUDA Support Test")
    print("=" * 60)
    
    if info['error']:
        print(f"❌ Error: {info['error']}")
        return
    
    print(f"✓ JAX Available: {info['jax_available']}")
    print(f"  JAX Version: {info['jax_version']}")
    print(f"  Default Backend: {info['default_backend']}")
    print(f"  Device Count: {info['device_count']}")
    
    if info['cuda_available']:
        print(f"✓ CUDA/GPU Available: Yes")
    else:
        print(f"✗ CUDA/GPU Available: No")
    
    print(f"\nDevices:")
    for dev_info in info['device_info']:
        print(f"  [{dev_info['id']}] {dev_info['device']}")
        print(f"      Platform: {dev_info['platform']}, Kind: {dev_info['device_kind']}")
    
    if 'computation_test' in info:
        print(f"\nComputation Test: {info['computation_test']}")
    
    print("=" * 60)


def test_package_imports() -> Dict[str, Dict[str, any]]:
    """
    Test all imports from the jaxcont package and its submodules.
    
    Returns:
        Dictionary mapping module names to their import status and details.
    """
    results = {}
    
    # Discover the installed package tree instead of maintaining a second,
    # easily-stale list of modules here.  In particular, the old continuation
    # classes were consolidated into core.scan_continuation in v0.2.
    package = importlib.import_module("jaxcont")
    modules_to_test = [
        package.__name__,
        *(module.name for module in pkgutil.walk_packages(
            package.__path__, prefix=f"{package.__name__}."
        )),
    ]
    
    for module_name in modules_to_test:
        result = {
            'success': False,
            'error': None,
            'attributes': []
        }
        
        try:
            module = importlib.import_module(module_name)
            result['success'] = True
            
            # Get public attributes (not starting with _)
            if hasattr(module, '__all__'):
                result['attributes'] = module.__all__
            else:
                result['attributes'] = [attr for attr in dir(module) if not attr.startswith('_')]
            
        except ImportError as e:
            result['error'] = f'ImportError: {str(e)}'
        except Exception as e:
            result['error'] = f'Error: {str(e)}'
        
        results[module_name] = result
    
    return results


def print_package_import_test():
    """Print formatted results of package import tests."""
    results = test_package_imports()
    
    print("=" * 60)
    print("JaxCont Package Import Test")
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    print(f"\nOverall: {success_count}/{total_count} modules imported successfully")
    print("=" * 60)
    
    # Print successful imports
    print("\n✓ Successful Imports:")
    for module_name, result in sorted(results.items()):
        if result['success']:
            print(f"  ✓ {module_name}")
            if result['attributes'] and len(result['attributes']) <= 10:
                print(f"    Exports: {', '.join(result['attributes'][:10])}")
            elif result['attributes']:
                print(f"    Exports: {len(result['attributes'])} items")
    
    # Print failed imports
    failed = {k: v for k, v in results.items() if not v['success']}
    if failed:
        print("\n✗ Failed Imports:")
        for module_name, result in sorted(failed.items()):
            print(f"  ✗ {module_name}")
            print(f"    {result['error']}")
    
    print("=" * 60)
    
    return success_count == total_count


def run_installation_tests():
    """Run all installation diagnostic tests (JAX/CUDA and package imports).
    
    This function runs comprehensive installation checks including:
    - JAX availability and version
    - CUDA/GPU support
    - All package module imports
    
    Note: This is different from unit tests in the tests/ directory.
    """
    print("\n" + "=" * 60)
    print("Running JaxCont Installation Tests")
    print("=" * 60 + "\n")
    
    # Test JAX and CUDA
    print_jax_cuda_info()
    print("\n")
    
    # Test package imports
    all_passed = print_package_import_test()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Check output above for details.")
    print("=" * 60 + "\n")
