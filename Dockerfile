# Base image: Now explicitly using CUDA 12.9.0 with cuDNN devel
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    wget \
    ca-certificates \
    # Added for faster compilation
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Set CUDA environment variables, to be available for PyTorch installation and X-Pose build
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Explicitly set CPATH for GCC to find CUDA headers
# This helps compilers find headers even if PATH isn't fully propagated.
ENV CPATH=${CUDA_HOME}/include:$CPATH
ENV CPLUS_INCLUDE_PATH=${CUDA_HOME}/include:$CPLUS_INCLUDE_PATH

# TORCH_CUDA_ARCH_LIST includes 8.0 and 8.6 for A100 compatibility
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# Install PyTorch, torchvision, and torchaudio versions compatible with CUDA 12.9
# You generally want the latest nightly/alpha build of PyTorch that matches the latest CUDA.
# As per NVIDIA's docs, PyTorch 2.7.0a0 is compatible with CUDA 12.9.
# The official PyTorch website might provide specific stable wheels for 12.x in the future.
# For now, using the nightly index might be necessary if stable cu129 wheels aren't out.
# Let's try to infer the correct URL based on PyTorch's usual structure for newer CUDA.
# This assumes PyTorch 2.4.0 is available for cu124. The latest is 2.7 for cu129.
# We need to install the correct PyTorch for CUDA 12.9
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 \
    # Remove the fixed version and rely on the nightly index if needed, OR:
    # If 2.7.0 is available:
    # pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu129

    # Given that PyTorch 2.3.1 was for cu121, and you're now on cuda 12.9,
    # it's better to get the latest PyTorch that is built for CUDA 12.9.
    # The nightly URL `https://download.pytorch.org/whl/nightly/cu129` or
    # `https://download.pytorch.org/whl/cu129` (if a stable release exists)
    # is what you would use. Since 2.7.0a0 is listed for cu129, you'd likely want to install that.
    # Let's use the nightly URL for cu129, and it will fetch the latest compatible.
    # If you need a specific 2.7.0a0 version, you might need to pin it.
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# The most reliable way to get a CUDA 12.9 compatible PyTorch would be to find the exact wheel:
# From your search results, a stable PyTorch for CUDA 12.9 is likely 2.7.0a0.
# Let's try to install that. Replace the above pip install with this:
# This assumes the nightly build for cu129 is what you need.
# As of current PyTorch 2.7.0a0 is matched with cu129 from NVIDIA.
RUN pip3 install --no-cache-dir "torch>=2.7.0.dev" "torchvision>=0.22.0.dev" "torchaudio>=2.7.0.dev" --pre --index-url https://download.pytorch.org/whl/nightly/cu129

# --- Python Diagnostic Script STARTS HERE ---
# Add this section to check PyTorch's CUDA status
RUN echo "Running CUDA and PyTorch diagnostics..." && \
    nvcc --version && \
    python3 <<EOF_PYTHON_SCRIPT
import torch
print(f'>>>> PyTorch version: {torch.__version__}')
_is_cuda_available = torch.cuda.is_available()
print(f'>>>> CUDA available for PyTorch: {_is_cuda_available}')
if _is_cuda_available:
    print(f'>>>> PyTorch CUDA version: {torch.version.cuda}')
    _device_count = torch.cuda.device_count()
    print(f'>>>> CUDA devices count: {_device_count}')
    if _device_count > 0:
        print(f'>>>> Current CUDA device: {torch.cuda.current_device()}')
        print(f'>>>> Device name: {torch.cuda.get_device_name(0)}')
    else:
        print('>>>> No CUDA devices found by PyTorch (though CUDA is reported as available).')
    print(f'>>>> torch.utils.cpp_extension.CUDA_HOME from PyTorch: {torch.utils.cpp_extension.CUDA_HOME}')
else:
    print('>>>> CUDA *NOT* available to PyTorch.')
    if hasattr(torch.version, 'cuda') and torch.version.cuda is not None:
        print(f'>>>> PyTorch compiled with CUDA version: {torch.version.cuda}')
    else:
        print('>>>> PyTorch CUDA version attribute not found or is None.')
    if hasattr(torch.utils, 'cpp_extension') and torch.utils.cpp_extension is not None and hasattr(torch.utils.cpp_extension, 'CUDA_HOME') and torch.utils.cpp_extension.CUDA_HOME is not None:
        print(f'>>>> torch.utils.cpp_extension.CUDA_HOME: {torch.utils.cpp_extension.CUDA_HOME}')
    else:
        print('>>>> torch.utils.cpp_extension.CUDA_HOME not found or is None.')
print('>>>> End of diagnostics <<<<')
EOF_PYTHON_SCRIPT
# --- Python Diagnostic Script ENDS HERE ---

# Copy the requirements files BEFORE trying to install them
# (Ensure requirements.txt and requirements_base.txt are in the build context)
COPY requirements.txt .
COPY requirements_base.txt .

# Install Python dependencies
# 1. Install from your custom requirements_base.txt (if any)
RUN if [ -f requirements_base.txt ]; then pip3 install --no-cache-dir -r requirements_base.txt; fi
# 2. Install from the main LivePortrait requirements.txt (as per readme)
RUN pip3 install --no-cache-dir -r requirements.txt
# 3. Install/Override specific packages as needed
# Now that base image is cuDNN 9, onnxruntime-gpu 1.21.0 might work.
# If it fails with cuDNN errors, downgrade to 1.17.0 again.
RUN pip3 install --no-cache-dir onnxruntime-gpu==1.21.0
RUN pip3 install --no-cache-dir transformers==4.38.0    # Pinned version
RUN pip3 install --no-cache-dir git+https://github.com/XPixelGroup/BasicSR.git
RUN pip3 install --no-cache-dir git+https://github.com/xinntao/Real-ESRGAN.git

# Copy the rest of the application's source code
COPY . .

# Download pretrained weights from HuggingFace
# Ensure the target directory exists
RUN mkdir -p ./pretrained_weights && \
    huggingface-cli download KwaiVGI/LivePortrait --local-dir ./pretrained_weights --exclude "*.git*" "README.md" "docs" --local-dir-use-symlinks False

# Build and install X-Pose dependency with proper GPU support (needed for Animals mode)
# This path is from the original LivePortrait readme.md for building X-Pose op.
RUN cd src/utils/dependencies/XPose/models/UniPose/ops && \
    # CUDA_HOME is already set as ENV, and CPATH/CPLUS_INCLUDE_PATH added for compilers.
    MAX_JOBS=1 python3 setup.py build install && \
    cd /app

# Make port 7860 available (Gradio default port)
EXPOSE 7860

# Set environment variable for Gradio server
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Define the command to run the application (Animals mode app)
CMD ["python3", "app_animals.py"]
