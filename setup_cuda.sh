#!/bin/bash
# Setup CUDA libraries for JAX

# Get Python site-packages directory
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Add all NVIDIA library paths
export LD_LIBRARY_PATH="\
$SITE_PACKAGES/nvidia/cublas/lib:\
$SITE_PACKAGES/nvidia/cuda_cupti/lib:\
$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:\
$SITE_PACKAGES/nvidia/cuda_runtime/lib:\
$SITE_PACKAGES/nvidia/cudnn/lib:\
$SITE_PACKAGES/nvidia/cufft/lib:\
$SITE_PACKAGES/nvidia/cusolver/lib:\
$SITE_PACKAGES/nvidia/cusparse/lib:\
$SITE_PACKAGES/nvidia/nccl/lib:\
$SITE_PACKAGES/nvidia/nvjitlink/lib:\
$SITE_PACKAGES/nvidia/nvshmem/lib:\
$LD_LIBRARY_PATH"

echo "CUDA libraries added to LD_LIBRARY_PATH"
echo "Testing JAX GPU access..."
python3 -c "import jax; print('JAX version:', jax.__version__); print('JAX devices:', jax.devices()); print('Default backend:', jax.default_backend())"
