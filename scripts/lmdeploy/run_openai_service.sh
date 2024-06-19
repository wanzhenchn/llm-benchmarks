#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gamil.com
# @Date     : 2024-04-15 16:19:07
# @Details  : Serving LLM with OpenAI Compatible Server
################################################################################

set -euxo pipefail

if [ $# != 4 ]; then
  echo "Usage: $0 model_path precision(fp16, w4a16 or kv8) port device_id(0 or 0,1 or 1,3)"
  exit
fi

model_path=$1
precision=$2
device_id=$4
gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")

export CUDA_VISIBLE_DEVICES=${device_id}

service_name=0.0.0.0
service_port=$3

# https://github.com/InternLM/lmdeploy/blob/main/docs/en/serving/api_server.md
# https://github.com/InternLM/lmdeploy/blob/main/docs/en/serving/api_server_vl.md
# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/cli/serve.py
if [ ! -e "gemm_config.in" ]; then
  echo "generate gemm_config.in now..."
  python3 -c "from lmdeploy.turbomind.generate_gemm_config import main;\
    main(tensor_para_size=${gpu_num}, max_batch_size=4, model_path='${model_path}')"
fi

if [ $precision = fp16 ]; then
  lmdeploy serve api_server \
    ${model_path} \
    --server-name ${service_name} \
    --server-port ${service_port} \
    --tp ${gpu_num} \
    --cache-max-entry-count 0.9 \
    --session-len 4096
#    --log-level INFO

elif [ $precision = kv8 ]; then
  lmdeploy serve api_server \
    ${model_path} \
    --server-name ${service_name} \
    --server-port ${service_port} \
    --tp ${gpu_num} \
    --cache-max-entry-count 0.9 \
    --session-len 4096 \
    --quant-policy 8 \
    --model-format hf

elif [ $precision = w4a16 ]; then
  lmdeploy serve api_server \
    ${model_path} \
    --server-name ${service_name} \
    --server-port ${service_port} \
    --tp ${gpu_num} \
    --cache-max-entry-count 0.9 \
    --session-len 4096 \
    --model-format awq
else
  echo "Precision only supports fp16, w4a16 or kv8"
  exit
fi
