#!/usr/bin/env bash
# Source this file after installing the offline package.
export STACK_ROOT="${STACK_ROOT:-/mnt/public/xzxuan}"
source /opt/venv/reason/bin/activate
# Core remains the image's original source installation.
export MEGATRON_PATH="/opt/venv/reason/Megatron-LM"
case ":${PYTHONPATH:-}:" in
    *":$MEGATRON_PATH:"*) ;;
    *) export PYTHONPATH="$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
# Ray channels maintain peer connections across the rollout workers.
if [[ "$(ulimit -Sn)" != unlimited ]] && (( $(ulimit -Sn) < 65536 )); then
    ulimit -Sn 65536
fi
export RLINF_ROOT="${RLINF_ROOT:-$STACK_ROOT/repos/RLinf}"
export RPENT_ROOT="${RPENT_ROOT:-$STACK_ROOT/repos/RPent}"
export RPENT_RLINF_ROOT="$RLINF_ROOT"
export RPENT_REPO_ROOT="$RPENT_ROOT"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
# The prepared Qwen3.8 template defaults to xhigh; omit the unsupported API tier.
export RPENT_REASONING_EFFORT="${RPENT_REASONING_EFFORT:-none}"
export NCCL_MAX_CTAS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$STACK_ROOT/envs/runtime-cache/huggingface"
export XDG_CACHE_HOME="$STACK_ROOT/envs/runtime-cache/xdg"
export TORCH_EXTENSIONS_DIR="$STACK_ROOT/envs/runtime-cache/torch-extensions"
export MODEL_QWEN35_4B="$STACK_ROOT/models/Qwen3.5-4B"
export MODEL_SFT595_4B="$STACK_ROOT/models/qwen3.5-4b-sft595-e3"
export MODEL_QWEN38_27B="$STACK_ROOT/models/Qwen3.8-27B"
export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
export LIBERO_TYPE=pro
export LIBERO_CONFIG_PATH="$STACK_ROOT/data/libero-pro/config"
export LIBERO_PRO_ASSET_PATH="$STACK_ROOT/data/libero-pro/assets"
export RPENT_TRAIN_TASKS="$STACK_ROOT/data/grpo-sft595/tasks/train.jsonl"
export RPENT_EVAL_TASKS="$STACK_ROOT/data/grpo-sft595/tasks/eval.jsonl"
