#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-11-01 10:32:15
# @Details  : launch openai service
################################################################################
set -euxo pipefail

if [ $# -ne 3 ] && [ $# -ne 4 ]; then
  echo "Usage: $0 model_path port (extra_args) gpu_device_id(0,1)"
  exit 1
fi

# https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/commands/serve.py

model_path=$1
port=$2
extra_args=${3-""}
device_id=${!#}

export CUDA_VISIBLE_DEVICES=${device_id}

tp=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")

trtllm-serve ${model_path} \
  --tokenizer ${model_path} \
  --port ${port} \
  --tp_size ${tp} \
  --kv_cache_free_gpu_memory_fraction 0.9 \
  --trust_remote_code ${extra_args}
