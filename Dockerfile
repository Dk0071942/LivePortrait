# Base image: Shifting to CUDA 11.8 with cuDNN 8 (very broad compatibility)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get
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
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Set CUDA environment variables, to be available for PyTorch installation and custom C++/CUDA extension builds
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# Explicitly set CPATH and CPLUS_INCLUDE_PATH for GCC/G++ to find CUDA headers
ENV CPATH=${CUDA_HOME}/include:$CPATH
ENV CPLUS_INCLUDE_PATH=${CUDA_HOME}/include:$CPLUS_INCLUDE_PATH

# TORCH_CUDA_ARCH_LIST includes relevant architectures for modern NVIDIA GPUs
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9"

# Install PyTorch, torchvision, and torchaudio for CUDA 11.8
RUN pip3 install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# --- Python Diagnostic Script STARTS HERE ---
RUN echo "Running CUDA and PyTorch diagnostics..." && \
    nvcc --version && \
    export CUDA_HOME=/usr/local/cuda && \
    echo ">>>> CUDA_HOME from shell: $CUDA_HOME" && \
    python3 <<EOF_PYTHON_SCRIPT
import os
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
    print(f'>>>> os.environ.get("CUDA_HOME"): {os.environ.get("CUDA_HOME")}')
    if hasattr(torch.utils, 'cpp_extension') and torch.utils.cpp_extension is not None and hasattr(torch.utils.cpp_extension, 'CUDA_HOME') and torch.utils.cpp_extension.CUDA_HOME is not None:
        print(f'>>>> torch.utils.cpp_extension.CUDA_HOME: {torch.utils.cpp_extension.CUDA_HOME}')
    else:
        print('>>>> torch.utils.cpp_extension.CUDA_HOME not found or is None.')
print('>>>> End of diagnostics <<<<')
EOF_PYTHON_SCRIPT
# --- Python Diagnostic Script ENDS HERE ---

# Copy the requirements files BEFORE trying to install them
COPY requirements.txt .
COPY requirements_base.txt .

# Install Python dependencies
RUN if [ -f requirements_base.txt ]; then pip3 install --no-cache-dir -r requirements_base.txt; fi
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir onnxruntime-gpu==1.16.2
RUN pip3 install --no-cache-dir transformers==4.38.0
RUN pip3 install --no-cache-dir git+https://github.com/XPixelGroup/BasicSR.git
RUN pip3 install --no-cache-dir git+https://github.com/xinntao/Real-ESRGAN.git

# Copy the rest of the application's source code
COPY . .

# Download pretrained weights from HuggingFace
RUN mkdir -p ./pretrained_weights && \
    huggingface-cli download KwaiVGI/LivePortrait --local-dir ./pretrained_weights --exclude "*.git*" "README.md" "docs" --local-dir-use-symlinks False

# Build and install X-Pose dependency with proper GPU support (needed for Animals mode)
# Explicitly ensuring CUDA_HOME and other critical variables are passed to setup.py's environment.
# Note: TORCH_CUDA_ARCH_LIST is already set as an ENV variable above.
RUN cd src/utils/dependencies/XPose/models/UniPose/ops && \
    echo "--- Attempting to build XPose UniPose ops with CUDA support (FORCE_CUDA=1) ---" && \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    FORCE_CUDA=1 \
    MAX_JOBS=1 python3 setup.py build_ext --verbose build install && \
    echo "--- XPose UniPose ops build finished ---" && \
    cd /app

# Make port 7860 available (Gradio default port)
EXPOSE 7860

# Set environment variable for Gradio server
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Define the command to run the application (Animals mode app)
CMD ["python3", "app_animals.py"]
