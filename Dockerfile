# SAC Meta-World Training Docker Image
# Optimized for DataLAB Cluster with GPU support

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies for MuJoCo and headless rendering
RUN apt-get update && apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    patchelf \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for headless rendering
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV DISPLAY=

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /workspace/rl_project_sac

# Set working directory to project
WORKDIR /workspace/rl_project_sac

# Create directories for logs and models
RUN mkdir -p logs models_mt10

# Set default command
CMD ["bash"]
