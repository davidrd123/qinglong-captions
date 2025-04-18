#!/usr/bin/env bash
set -e  # Exit on error

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading API keys from .env file"
    set -a  # automatically export all variables
    source .env
    set +a
else
    echo "No .env file found. Please create one with your API keys"
    echo "Example .env file:"
    echo "GEMINI_API_KEY=your-key-here"
    echo "PIXTRAL_API_KEY=your-key-here"
    echo "STEP_API_KEY=your-key-here"
    echo "QWENVL_API_KEY=your-key-here"
fi

# Configuration variables
dataset_path="./datasets/hidari_ben_clips_v2"  # Default path
if [ $# -gt 0 ]; then
    dataset_path="$1"  # Override with command-line argument if provided
    echo "Using provided dataset path: $dataset_path"
else
    echo "Using default dataset path: $dataset_path"
    echo ""
    echo "To use a different path, run:"
    echo "  ./run.sh path/to/dataset"
    echo "  Example: ./run.sh ./datasets/project1"
    echo ""
    echo "Available datasets:"
    if [ -d "./datasets" ]; then
        ls -1 ./datasets/
    else
        echo "No datasets directory found"
    fi
fi

# Verify dataset path exists
if [ ! -d "$dataset_path" ]; then
    echo "Error: Dataset path '$dataset_path' not found"
    exit 1
fi

# Configuration variables
gemini_model_path="gemini-2.5-pro-preview-03-25"
# gemini_model_path="gemini-2.0-flash"
pixtral_model_path="pixtral-large-2411"
step_model_path="step-1.5v-mini"
qwenVL_model_path="qwen-vl-max-latest" # qwen2.5-vl-72b-instruct<10mins qwen-vl-max-latest <1min
dir_name=true
mode="long"
not_clip_with_caption=false
wait_time=1
max_retries=100
segment_time=300

# Environment variables
export HF_HOME="huggingface"
export XFORMERS_FORCE_DISABLE_TRITON=1
export PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG=1

# Initialize arguments array
ext_args=()

# Build arguments array based on configuration
[ -n "$GEMINI_API_KEY" ] && ext_args+=("--gemini_api_key=$GEMINI_API_KEY")
[ -n "$PIXTRAL_API_KEY" ] && ext_args+=("--pixtral_api_key=$PIXTRAL_API_KEY")
[ -n "$STEP_API_KEY" ] && ext_args+=("--step_api_key=$STEP_API_KEY")
[ -n "$QWENVL_API_KEY" ] && ext_args+=("--qwenVL_api_key=$QWENVL_API_KEY")
[ -n "$gemini_model_path" ] && ext_args+=("--gemini_model_path=$gemini_model_path")
[ -n "$pixtral_model_path" ] && ext_args+=("--pixtral_model_path=$pixtral_model_path")
[ -n "$step_model_path" ] && ext_args+=("--step_model_path=$step_model_path")
[ -n "$qwenVL_model_path" ] && ext_args+=("--qwenVL_model_path=$qwenVL_model_path")

# Add boolean and value flags
[ "$dir_name" = true ] && ext_args+=("--dir_name")
[ "$mode" != "all" ] && ext_args+=("--mode=$mode")
[ "$not_clip_with_caption" = true ] && ext_args+=("--not_clip_with_caption")
[ "$wait_time" -ne 1 ] && ext_args+=("--wait_time=$wait_time")
[ "$max_retries" -ne 20 ] && ext_args+=("--max_retries=$max_retries")
[ "$segment_time" -ne 300 ] && ext_args+=("--segment_time=$segment_time")

# Debug the arguments
echo "Debug: ext_args contents:"
printf '%s\n' "${ext_args[@]}"

# Activate virtual environment
if [ -d "./venv/bin" ]; then
    echo "Activating venv"
    source ./venv/bin/activate
elif [ -d "./.venv/bin" ]; then
    echo "Activating .venv"
    source ./.venv/bin/activate
else
    echo "No virtual environment found"
    exit 1
fi

# Debug: Print environment variables
echo "Environment variables:"
echo "GEMINI_API_KEY=${GEMINI_API_KEY:0:5}..." # Only show first 5 chars for security
echo "PIXTRAL_API_KEY=${PIXTRAL_API_KEY:0:5}..."
echo "STEP_API_KEY=${STEP_API_KEY:0:5}..."
echo "QWENVL_API_KEY=${QWENVL_API_KEY:0:5}..."

# Run captioner with debug
echo "Running command: python -m module.captioner $dataset_path ${ext_args[@]}"
python -m module.captioner "$dataset_path" "${ext_args[@]}"

echo "Captioner finished"
read -p "Press Enter to exit"
