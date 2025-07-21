# GPU Dependencies Fix for LivePortrait Animal Mode

## Problem Summary

LivePortrait Animal Mode (`app_animals.py`) fails to launch due to missing GPU dependencies for the X-Pose extension. The system has CUDA runtime but lacks the CUDA development toolkit needed to compile the `MultiScaleDeformableAttention` extension.

## Root Cause

- **CUDA Runtime vs Development Toolkit**: System had CUDA runtime libraries but no development toolkit
- **Missing X-Pose Extension**: `MultiScaleDeformableAttention` module failed to compile
- **Environment Variables**: `CUDA_HOME` was `None`, PyTorch couldn't find CUDA headers
- **Library Path Issues**: Missing PyTorch libraries in `LD_LIBRARY_PATH`

## Quick Fix

### 1. Install CUDA Development Toolkit

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit 12.1 (matches PyTorch 2.3.0+cu121)
sudo apt-get install cuda-toolkit-12-1 -y
```

### 2. Set Environment Variables

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Make permanent
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Rebuild X-Pose Extension

```bash
conda activate LivePortrait
cd src/utils/dependencies/XPose/models/UniPose/ops

# Clean previous build
rm -rf build/ dist/ *.egg-info

# Set environment and rebuild with CUDA
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

FORCE_CUDA=1 python setup.py build install
```

### 4. Fix Runtime Library Path

```bash
# Add PyTorch libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/home/ubuntu/miniconda3/envs/LivePortrait/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### 5. Launch Animal Mode

```bash
# Set all environment variables
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/home/ubuntu/miniconda3/envs/LivePortrait/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# Launch app
python app_animals.py --server_name 0.0.0.0 --server_port 7860
```

## Verification Steps

1. **Check CUDA compiler**: `nvcc --version` should show CUDA 12.1
2. **Verify PyTorch CUDA**: `python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"` should show `/usr/local/cuda-12.1`
3. **Test X-Pose module**: `python -c "import MultiScaleDeformableAttention; print('SUCCESS!')"`
4. **Launch animal mode**: `python app_animals.py` should start without errors

## Common Issues

### `nvcc: command not found`
- **Solution**: Install CUDA toolkit and add to PATH

### `CUDA_HOME: None`
- **Solution**: Set `CUDA_HOME` environment variable

### `libc10.so: cannot open shared object file`
- **Solution**: Add PyTorch lib directory to `LD_LIBRARY_PATH`

### Compilation warnings
- **Solution**: Warnings are normal, compilation should still succeed

## Automation Script

```bash
#!/bin/bash
# setup_gpu_animal_mode.sh

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1 -y

# Set environment variables
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Make permanent
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Rebuild X-Pose
conda activate LivePortrait
cd src/utils/dependencies/XPose/models/UniPose/ops
rm -rf build/ dist/ *.egg-info
FORCE_CUDA=1 python setup.py build install

echo "Animal mode setup complete!"
```

## Key Points

- **CUDA Runtime ≠ Development Toolkit**: Runtime runs CUDA apps, development toolkit compiles them
- **Version Matching**: CUDA toolkit version must match PyTorch CUDA version
- **Environment Variables**: `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH` must be set correctly
- **Library Dependencies**: Both CUDA and PyTorch libraries needed in `LD_LIBRARY_PATH`

## Result

✅ **app_animals.py** launches successfully with GPU acceleration  
✅ **X-Pose MultiScaleDeformableAttention** compiled with CUDA support  
✅ **Animal mode** ready for use with full GPU performance