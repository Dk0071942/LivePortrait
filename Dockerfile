# Base image: Ubuntu 22.04 with CUDA 12.1
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

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
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Set CUDA environment variables first, to be available for PyTorch installation and X-Pose build
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install specific PyTorch, torchvision, and torchaudio versions
# Note: This line was changed in your latest logs to torch 2.7.0 / cu128.
# Ensure this is the version you intend to use.
# Install specific PyTorch, torchvision, and torchaudio versions for CUDA 12.1
RUN pip3 install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# --- Python Diagnostic Script STARTS HERE ---
# Add this section to check PyTorch's CUDA status
RUN echo "Running CUDA and PyTorch diagnostics..." && nvcc --version && python3 <<EOF_PYTHON_SCRIPT
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
    # Attempt to print CUDA version string from PyTorch even if not available
    if hasattr(torch.version, 'cuda') and torch.version.cuda is not None:
        print(f'>>>> PyTorch compiled with CUDA version: {torch.version.cuda}')
    else:
        print('>>>> PyTorch CUDA version attribute not found or is None.')
    # Attempt to print CUDA_HOME from cpp_extension
    if hasattr(torch.utils, 'cpp_extension') and torch.utils.cpp_extension is not None and hasattr(torch.utils.cpp_extension, 'CUDA_HOME') and torch.utils.cpp_extension.CUDA_HOME is not None:
        print(f'>>>> torch.utils.cpp_extension.CUDA_HOME: {torch.utils.cpp_extension.CUDA_HOME}')
    else:
        print('>>>> torch.utils.cpp_extension.CUDA_HOME not found or is None.')
print('>>>> End of diagnostics <<<<')
EOF_PYTHON_SCRIPT
# --- Python Diagnostic Script ENDS HERE ---

# Copy the requirements files BEFORE trying to install them
COPY requirements.txt .
# If you also have requirements_base.txt and it's needed, copy it too.
COPY requirements_base.txt .

# Install Python dependencies from requirements.txt
# It's good practice to keep --no-cache-dir to reduce image size
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code
COPY . .

# Download pretrained weights from HuggingFace
# Ensure the target directory exists
RUN mkdir -p ./pretrained_weights && \
    huggingface-cli download KwaiVGI/LivePortrait --local-dir ./pretrained_weights --exclude "*.git*" "README.md" "docs" --local-dir-use-symlinks False

# Add an explicit check for CUDA availability before attempting X-Pose build
RUN python3 -c "import torch; print(f'>>>> Pre-XPose build check: torch.cuda.is_available() is {torch.cuda.is_available()}'); assert torch.cuda.is_available(), 'CUDA must be available to PyTorch for X-Pose build. Check Docker host GPU/NVIDIA runtime configuration.'"

# Build and install X-Pose dependency
RUN cd src/utils/dependencies/XPose/models/UniPose/ops && \
    python3 setup.py build install && \
    cd /app

# Make port 7860 available (Gradio default port)
EXPOSE 7860

# Set environment variable for Gradio server
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Define the command to run the application
CMD ["python3", "app_animals.py"]
