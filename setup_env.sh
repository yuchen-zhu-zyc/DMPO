#!/bin/bash
set -e

ENV_NAME="${1:-.venv}"

echo "=== PCE-MDM Environment Setup ==="
echo "Target venv: ${ENV_NAME}"

# Check if uv is installed; if not, install it
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv version: $(uv --version)"

# Create virtual environment with Python 3.10
echo "Creating virtual environment with Python 3.10..."
uv venv --python 3.10 "${ENV_NAME}"

# Activate the venv for the rest of this script
source "${ENV_NAME}/bin/activate"

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
uv pip install \
    torch==2.6.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install core ML dependencies
echo "Installing core dependencies..."
uv pip install \
    transformers==4.49.0 \
    accelerate==1.4.0 \
    bitsandbytes==0.45.3 \
    peft==0.15.1 \
    deepspeed==0.16.4 \
    datasets==3.3.2

# Install TRL from the specific commit used in this project
echo "Installing TRL (pinned commit)..."
uv pip install "git+https://github.com/huggingface/trl.git@0f88c179e30b3439467942a08c3190f624d5c423"

# Install numerical / data libraries
echo "Installing numerical libraries..."
uv pip install \
    "numpy==1.25.0" \
    pandas \
    scipy \
    scikit-learn

# Install remaining utilities
echo "Installing utilities..."
uv pip install \
    tiktoken==0.9.0 \
    wandb==0.15.3 \
    sentencepiece \
    evaluate \
    tqdm \
    regex \
    rich

# Install flash-attn (requires torch to be installed first)
echo "Installing flash-attn (this may take a few minutes)..."
uv pip install flash-attn --no-build-isolation

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with:"
echo "  source ${ENV_NAME}/bin/activate"
