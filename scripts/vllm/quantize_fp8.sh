#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-06-25 11:09:23
# @Details  : Offline FP8 Quantization with Static Activation Scaling Factors
################################################################################

set -euxo pipefail

# https://github.com/vllm-project/vllm/blob/main/docs/source/quantization/fp8.rst

if [ $# = 3 ]; then
  model_path=$1
  output_model_path=$2
  device_id=$3

  gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")
  export CUDA_VISIBLE_DEVICES=$device_id

  if [ "$(pip list | grep transformer-engine | wc -l)" -ne "0" ]; then
    pip uninstall -y transformer-engine
  fi

  if [ "$(pip list | grep llmcompressor | wc -l)" -eq "0" ]; then
    pip install llmcompressor
  fi

  if [ ! -d ${output_model_path} ]; then
    python quantize_fp8.py \
      --model_path ${model_path} \
      --saved_path ${output_model_path}
  else
    echo "The quantized hf model already exits in ${output_model_path}"
  fi

else
  echo "Usage: $0 hf_model_path quantized_hf_model_path device_id(0,1)"
  exit
fi
