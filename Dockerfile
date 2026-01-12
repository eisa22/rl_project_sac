# CUDA 12.1 + PyTorch + JAX + MuJoCo for MTMH-SAC
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libssl-dev \
    libffi-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Python dependencies (matching requirements.txt)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /source

# Expose port for jupyter if needed
EXPOSE 8888

# Default command
CMD ["/bin/bash"]
