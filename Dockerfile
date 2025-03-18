# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Set a working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    nvidia-container-toolkit\
    cuda-toolkit-12-6 \
    && rm -rf /var/lib/apt/lists/*


# Create a virtual environment (optional, but recommended)
RUN pip install --upgrade pip && \
    pip install virtualenv && \
    virtualenv venv && \
    . venv/bin/activate



# Install the package and its dependencies


# Install additional dependencies for CUDA and PyTorch
RUN pip install torch torchvision torchaudio 

RUN pip install h5py \
	openslide-python \
	openslide-bin \
	timm \
	shapely \
	subscriptable-path

RUN mkdir -p /app/tools
RUN mkdir -p /app/model
RUN mkdir -p /app/script


# Clone the instanseg-torch repository
RUN git clone https://github.com/instanseg/instanseg.git /app/tools/instanseg/


WORKDIR /app/tools/instanseg/
RUN pip install -e .  # Installs the package in editable mode


COPY ExtractFeatures/ /app/tools/FeaturesCellsExtractor/ 

COPY pytorch_model.bin /app/model
COPY run_all.py /app/script

RUN mkdir -p /app/images


WORKDIR /app/
# Set entrypoint to bash (or modify as needed)
ENTRYPOINT ["/bin/bash"]

