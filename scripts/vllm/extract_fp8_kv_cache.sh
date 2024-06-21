#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-02 17:09:23
# @Details  : Extracts the KV cache scaling factors from a quantized HF model
################################################################################

set -euxo pipefail

# https://github.com/vllm-project/vllm/tree/main/examples/fp8

if [ $# = 4 ]; then
  model_path=$1
  output_model_path=$2
  output_kv_cache_path=$3
  device_id=$4

  gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")
  export CUDA_VISIBLE_DEVICES=$device_id

  pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-ammo==0.7.1

  # 1. Convert HF model into a quantized HF model.
  if [ ! -d ${output_model_path} ]; then
    python fp8/quantize.py \
      --model_dir ${model_path} \
      --dtype float16 \
      --qformat fp8 \
      --kv_cache_dtype fp8 \
      --calib_size 512 \
      --tp_size ${gpu_num} \
      --output_dir ${output_model_path}
  else
    echo "The quantized hf model already exits in ${output_model_path}"
  fi

  # 2. Extract KV Cache Scaling Factors from quantized HF model.
  python3 fp8/extract_scales.py \
    --quantized_model ${output_model_path} \
    --tp_size ${gpu_num} \
    --output_name "kv_cache_fp8_scales.json" \
    --output_dir $output_kv_cache_path

else
  echo "Usage: $0 hf_model_path quantized_hf_model_path kv_cache_path device_id(0,1)"
  exit
fi
