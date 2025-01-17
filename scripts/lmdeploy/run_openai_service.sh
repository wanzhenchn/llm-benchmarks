#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gamil.com
# @Date     : 2024-04-15 16:19:07
# @Details  : Serving LLM with OpenAI Compatible Server
################################################################################

set -euxo pipefail

if [ $# != 4 ]; then
  echo "Usage: $0 model_path precision(fp16, w4a16 or kv8, fp8) port device_id(0 or 0,1 or 1,3)"
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

extra_args=""

if [ $precision = fp16 ] || [ $precision = w4a16 ] || \
   [ $precision = fp16-kv-int8 ] || [ $precision = fp16-kv-fp8 ] || \
   [ $precision = fp8-kv-fp16 ] || [ $precision = fp8-kv-fp8 ]; then

  if [ $precision = w4a16 ]; then
    # AWQ + KV-INT8
    extra_args+="--model-format awq "
    extra_args+="--quant-policy 8 "
  elif [ $precision = fp16-kv-int8 ]; then
    # FP16 + KV-INT8
    extra_args+="--model-format hf "
    extra_args+="--quant-policy 8 "

  elif [ $precision = fp16-kv-fp8 ]; then
    # FP16 + KV-FP8
    extra_args+="--model-format hf "
    extra_args+="--quant-policy 16 "

  elif [ $precision = fp8-kv-fp16 ]; then
    # FP8 + KV-FP16
    extra_args+="--model-format fp8 "
    extra_args+="--quant-policy 0 "

  elif [ $precision = fp8-kv-fp8 ]; then
    # FP8 + KV-FP8
    extra_args+="--model-format fp8 "
    extra_args+="--quant-policy 16 "
  fi

  lmdeploy serve api_server \
    ${model_path} \
    --server-name ${service_name} \
    --server-port ${service_port} \
    --tp ${gpu_num} \
    --cache-max-entry-count 0.9 \
    --session-len 16384 \
    --max-batch-size 256 ${extra_args} --max-log-len 0
#    --enable-prefix-caching \

else
  echo "precision only support fp16, w4a16, fp16-kv-int8, fp16-kv-fp8, fp8-kv-fp16, fp8-kv-fp8"
  exit
fi
