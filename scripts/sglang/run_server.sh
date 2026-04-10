#! /usr/bin/bash

set -euxo pipefail

export PYTHONPATH=/opt/sglang/python
export SGLANG_DISABLE_CUDNN_CHECK=1

if [ $# -lt 4 ]; then
  echo "Usage: $0 <port> <model_path> <device_id> <enable_atom>"
  exit 1
fi

port=$1
model_path=$2
device_id=$3
enable_atom=$4 # true or false

if [ "${enable_atom}" == true ]; then
  export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
fi

tp=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")


HIP_VISIBLE_DEVICES=${device_id} python3 -m sglang.launch_server \
  --model-path ${model_path} \ 
  --port ${port} \
  --tensor-parallel-size ${tp} \ 
  --mem-fraction-static 0.9 \
  --reasoning-parser qwen3 \
  --disable-radix-cache
  # --context-length 61440 
  # --page-size 1 --disable-cuda-graph  --cuda-graph-max-bs 16
  # --enable-dp-attention --dp-size ${tp} --kv-cache-dtype fp8_e4m3 