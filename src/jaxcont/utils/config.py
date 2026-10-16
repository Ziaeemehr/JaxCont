"""Installation diagnostics for JaxCont."""

import importlib
import pkgutil


def test_jax_cuda() -> dict[str, any]:
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
            jnp.sum(x)
            result['computation_test'] = 'passed'
        except Exception as e:
            result['computation_test'] = f'failed: {e!s}'
            
    except ImportError as e:
        result['error'] = f'JAX import failed: {e!s}'
    except Exception as e:
        result['error'] = f'Unexpected error: {e!s}'
    
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
        print("✓ CUDA/GPU Available: Yes")
    else:
        print("✗ CUDA/GPU Available: No")
    
    print("\nDevices:")
    for dev_info in info['device_info']:
        print(f"  [{dev_info['id']}] {dev_info['device']}")
        print(f"      Platform: {dev_info['platform']}, Kind: {dev_info['device_kind']}")
    
    if 'computation_test' in info:
        print(f"\nComputation Test: {info['computation_test']}")
    
    print("=" * 60)


def test_package_imports() -> dict[str, dict[str, any]]:
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
            result['error'] = f'ImportError: {e!s}'
        except Exception as e:
            result['error'] = f'Error: {e!s}'
        
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
