#!/bin/bash
# Common configuration for 2x RTX 3090 SGLang inference
#
# SGLang v0.5.11 + local patches (see patches/README.md). NCCL for
# multi-GPU communication.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Conda ---
if [ -z "${CONDA_BASE:-}" ]; then
    if [ -n "${CONDA_EXE:-}" ]; then
        CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
    elif [ -d "$HOME/miniforge3" ]; then
        CONDA_BASE="$HOME/miniforge3"
    elif [ -d "$HOME/mambaforge" ]; then
        CONDA_BASE="$HOME/mambaforge"
    elif [ -d "$HOME/miniconda3" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif command -v conda &>/dev/null; then
        CONDA_BASE="$(conda info --base 2>/dev/null)"
    else
        # HOST ADAPTATION (/dd2/sglang): this machine has no conda — the serving env is a
        # uv venv at /dd2/sglang/v0520, activated by run_qwen38.sh before launch. When a
        # venv is already active we proceed without conda instead of aborting.
        if [ -n "${VIRTUAL_ENV:-}" ]; then
            export SGLANG_NO_CONDA=1
        else
            echo "ERROR: Cannot find conda. Set CONDA_BASE=/path/to/conda, or activate a venv (run_qwen38.sh)."
            exit 1
        fi
    fi
fi
export CONDA_BASE

# Default stack flipped to v0.5.20 (2026-09-19). The prior v0.5.18
# (/data/sglang-rebase-v0518, env `sglang-v0518`), v0.5.17 (/data/sglang-rebase-v0517,
# env `sglang-v0517`) and older trees (v0.5.16/v0.5.15/v0.5.14, same naming) are
# retained untouched for one-revert rollback — revert this commit to fall back. Override
# ENV_NAME/SGLANG_DIR to run an older stack.
ENV_NAME="${ENV_NAME:-sglang-v0520}"
SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-v0520}"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
PORT="${PORT:-23334}"
BASE_URL="http://localhost:${PORT}"

init_conda() {
    # HOST ADAPTATION (/dd2/sglang): no-op without conda — see the SGLANG_NO_CONDA guard above.
    [ -n "${SGLANG_NO_CONDA:-}" ] && return 0
    eval "$($CONDA_BASE/bin/conda shell.bash hook)"
}

activate_conda() {
    [ -n "${SGLANG_NO_CONDA:-}" ] && return 0
    init_conda
    conda activate "$ENV_NAME"
}

# NVIDIA environment setup
setup_nvidia_env() {
    # GPU selection (both 3090s)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

    # NCCL P2P over PCIe
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0

    # NCCL debug — INFO so we can see transport selection
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,P2P}

    # PyTorch memory
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    export TOKENIZERS_PARALLELISM=false
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    export PYTHONWARNINGS="ignore::UserWarning"

    # Force tvm_ffi's JIT compiler to target CUDA. If /opt/rocm exists or
    # ROCM_PATH/ROCM_HOME is set (R9700 sister-stack leftovers on a swapped
    # motherboard, or a partial ROCm install for other reasons), tvm_ffi
    # auto-detects "hip" and crashes the awq_marlin_repack JIT with
    # `Could not detect ROCm GPU architecture automatically`.
    export TVM_FFI_GPU_BACKEND=cuda
}
