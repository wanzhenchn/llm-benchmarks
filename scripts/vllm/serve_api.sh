#!/usr/bin/env bash

set -euxo pipefail

# https://docs.vllm.ai/en/latest/models/engine_args.html
host_name=0.0.0.0
port=8000

if [ $# = 2 ]; then
  model_path=$1
  device_id=$2
  gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")

  CUDA_VISIBLE_DEVICES=$device_id \
  python3 -m vllm.entrypoints.api_server \
    --host ${host_name} \
    --port $port \
    --model ${model_path} \
    --dtype auto `# “auto” will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.`\
    -tp ${gpu_num} \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --disable-log-stats
#    --uvicorn-log-level info `# ['debug', 'info', 'warning', 'error', 'critical', 'trace']`
#    --quantization `# {awq,squeezellm,None}`

else
  echo "Usage1: $0 model_path device_id(0 or 0,1 or 1,3)"
  exit
fi
