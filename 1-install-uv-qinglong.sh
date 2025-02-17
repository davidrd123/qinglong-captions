#!/usr/bin/env bash
set -e  # Exit on error

# Environment Variables Setup
export HF_HOME="huggingface"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"
export UV_CACHE_DIR="${HOME}/.cache/uv"
export UV_NO_BUILD_ISOLATION=1
export UV_NO_CACHE=0
export UV_LINK_MODE="symlink"  # Better default for Unix
export GIT_LFS_SKIP_SMUDGE=1
export CUDA_HOME="${CUDA_PATH}"

# Error handling function
error_exit() {
    echo "Error: $1" >&2
    echo "Installation failed. Press any key to exit."
    read -n 1
    exit 1
}

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y libmediainfo-dev

# Get UV path
UV_PATH=$(which uv)

# Check for UV installation
if ! command -v uv &> /dev/null; then
    echo "Installing UV module..."
    curl -LsSf https://astral.sh/uv/0.4.1/install.sh | sh || error_exit "UV installation failed"
    UV_PATH=$(which uv)
else
    UV_VERSION=$(uv --version)
    echo "UV module is installed. Version: $UV_VERSION"
fi

# Create and activate virtual environment
if [ -d "./venv/bin" ]; then
    echo "Activating existing venv"
    source ./venv/bin/activate
elif [ -d "./.venv/bin" ]; then
    echo "Activating existing .venv"
    source ./.venv/bin/activate
else
    echo "Creating new .venv"
    "$UV_PATH" venv -p 3.10 || error_exit "Failed to create virtual environment"
    source ./.venv/bin/activate
fi

# Install packages
echo "Installing main requirements"
"$UV_PATH" pip install --upgrade setuptools wheel || error_exit "Failed to upgrade setuptools and wheel"
"$UV_PATH" pip sync requirements-uv.txt --index-strategy unsafe-best-match || error_exit "Failed to install requirements"

echo "Installation completed successfully"
read -p "Press Enter to exit"