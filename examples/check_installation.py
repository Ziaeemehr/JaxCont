#!/usr/bin/env python3
"""
Test JaxCont installation, JAX/CUDA support, and package imports.

This script provides comprehensive diagnostics for your JaxCont installation.
Note: This tests installation and environment setup, not unit tests (see tests/ directory).
"""

from jaxcont.utils import (
    print_jax_cuda_info,
    print_package_import_test,
    run_installation_tests,
    test_jax_cuda,
)


def main():
    """Run installation tests."""
    
    # Option 1: Run all installation tests at once
    print("Running comprehensive installation diagnostic tests...\n")
    run_installation_tests()
    
    # Option 2: Run individual tests
    # Uncomment the following to run tests separately:
    
    # # Test JAX and CUDA only
    # print_jax_cuda_info()
    
    # # Test package imports only
    # print_package_import_test()
    
    # # Get raw data (programmatic access)
    # cuda_info = test_jax_cuda()
    # if cuda_info['cuda_available']:
    #     print(f"\n✓ CUDA is available with {cuda_info['device_count']} device(s)")
    # else:
    #     print("\n✗ CUDA is not available. Running on CPU.")


if __name__ == "__main__":
    main()
