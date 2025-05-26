# Base image: Ubuntu 22.04 with CUDA 12.1
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - git: for version control (useful if cloning repos during build)
# - ffmpeg: for video processing
# - libgl1-mesa-glx: dependency for OpenCV and other graphics libraries
# - python3.10, python3-pip, python3.10-dev: Python runtime and development tools
# - build-essential: for compiling C/C++ code (e.g., X-Pose)
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

# Install specific PyTorch, torchvision, and torchaudio versions for CUDA 12.1
# This is done *after* CUDA ENV VARS are set.
RUN pip3 install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Copy the requirements files
COPY requirements.txt .
COPY requirements_base.txt .

# Install Python dependencies from requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code
COPY . .

# Download pretrained weights from HuggingFace
# Ensure the target directory exists
RUN mkdir -p ./pretrained_weights && \
    huggingface-cli download KwaiVGI/LivePortrait --local-dir ./pretrained_weights --exclude "*.git*" "README.md" "docs" --local-dir-use-symlinks False

# Build and install X-Pose dependency
# This is required for animals mode and potentially other functionalities
# CUDA ENV VARS should be picked up by torch.cuda.is_available() now.
RUN cd src/utils/dependencies/XPose/models/UniPose/ops && \
    python3 setup.py build install && \
    cd /app

# Make port 7860 available (Gradio default port)
EXPOSE 7860

# Set environment variable for Gradio server (already set by Gradio itself, but good practice)
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Define the command to run the application (animals mode Gradio interface)
CMD ["python3", "app_animals.py"]
