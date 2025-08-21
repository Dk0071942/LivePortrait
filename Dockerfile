# Base image: CUDA 12.2.2 with cuDNN 8 for better PyTorch compatibility
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

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
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install PyTorch, torchvision, and torchaudio for CUDA 12.1
RUN pip3 install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# --- Python Diagnostic Script STARTS HERE ---
# To ensure this runs and isn't cached from a potentially different context,
# add a cache-busting element like the current date.
RUN echo "Running CUDA and PyTorch diagnostics... $(date)" && \
    nvcc --version && \
    export CUDA_HOME_SHELL_EXPORT=/usr/local/cuda && \
    echo ">>>> CUDA_HOME from shell export: $CUDA_HOME_SHELL_EXPORT" && \
    echo ">>>> CUDA_HOME from ENV: $CUDA_HOME" && \
    echo ">>>> PATH from ENV: $PATH" && \
    echo ">>>> LD_LIBRARY_PATH from ENV: $LD_LIBRARY_PATH" && \
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
        print('>>>> No CUDA devices found by PyTorch (though CUDA is reported as available). This is OK for build if toolkit is fine.')
    print(f'>>>> torch.utils.cpp_extension.CUDA_HOME from PyTorch: {torch.utils.cpp_extension.CUDA_HOME}')
else:
    print('>>>> CUDA *NOT* available to PyTorch according to torch.cuda.is_available().')
    if hasattr(torch.version, 'cuda') and torch.version.cuda is not None:
        print(f'>>>> PyTorch compiled with CUDA version: {torch.version.cuda}')
    else:
        print('>>>> PyTorch CUDA version attribute not found or is None.')
    print(f'>>>> os.environ.get("CUDA_HOME") as seen by Python: {os.environ.get("CUDA_HOME")}')
    if hasattr(torch.utils, 'cpp_extension') and torch.utils.cpp_extension is not None and hasattr(torch.utils.cpp_extension, 'CUDA_HOME') and torch.utils.cpp_extension.CUDA_HOME is not None:
        print(f'>>>> torch.utils.cpp_extension.CUDA_HOME from PyTorch: {torch.utils.cpp_extension.CUDA_HOME}')
    else:
        print('>>>> torch.utils.cpp_extension.CUDA_HOME not found or is None (when torch.cuda.is_available() is False).')
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

# Download the RealESRGAN model for upscaling
RUN mkdir -p ./src/utils/upscale_models && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O ./src/utils/upscale_models/RealESRGAN_x4plus.pth

# Build and install X-Pose dependency with proper GPU support (needed for Animals mode)
RUN cd src/utils/dependencies/XPose/models/UniPose/ops && \
    echo "--- Current directory: $(pwd) ---" && \
    echo "--- Listing directory contents (ops): ---" && \
    ls -la && \
    echo "--- Environment check before nvcc: ---" && \
    echo "--- Initial PATH: $PATH" && \
    PYTHON3_EXEC_PATH=$(which python3) && \
    echo "--- Found python3 at: $PYTHON3_EXEC_PATH ---" && \
    if [ -z "$PYTHON3_EXEC_PATH" ]; then echo "CRITICAL: python3 not found in PATH" >&2; exit 1; fi && \
    echo "--- LD_LIBRARY_PATH: $LD_LIBRARY_PATH" && \
    echo "--- CUDA_HOME: $CUDA_HOME" && \
    echo "--- TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST" && \
    echo "--- Running nvcc --version directly: ---" && \
    nvcc --version && \
    NVCC_EXIT_CODE=$? && \
    echo "--- nvcc exit code: $NVCC_EXIT_CODE ---" && \
    if [ $NVCC_EXIT_CODE -ne 0 ]; then \
        echo "CRITICAL: nvcc command failed! CUDA toolkit is not correctly set up or accessible." >&2; \
        exit 1; \
    fi && \
    echo "--- nvcc check PASSED. Attempting to build XPose UniPose ops with CUDA support (FORCE_CUDA=1) ---" && \
    env CUDA_HOME=/usr/local/cuda \
        PATH="/usr/local/cuda/bin:$PATH" \
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH" \
        FORCE_CUDA=1 \
        MAX_JOBS=1 \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        "$PYTHON3_EXEC_PATH" setup.py build_ext --verbose build install && \
    echo "--- XPose UniPose ops build finished ---" && \
    cd /app

# Make port 8890 available (Gradio default port)
EXPOSE 8890

# Set environment variable for Gradio server
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Define the command to run the application (Animals mode app)
# Explicitly set server-name to 0.0.0.0 to accept connections from any interface
CMD ["python3", "app_animals.py", "--server-name", "0.0.0.0"]
