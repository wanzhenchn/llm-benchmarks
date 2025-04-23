#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-06-25 11:09:23
# @Details  : Apply fp8/awq Quantization on LLMs with llm-compressor
################################################################################

set -euxo pipefail

if [ $# = 4 ]; then
  model_path=$1
  quant_method=$2
  output_model_path=$3
  device_id=$4

  gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")
  export CUDA_VISIBLE_DEVICES=$device_id

  SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

  if [ "$(pip list | grep transformer-engine | wc -l)" -ne "0" ]; then
    pip3 uninstall -y transformer-engine
  fi

  if [ "$(pip list | grep llmcompressor | wc -l)" -eq "0" ]; then
    pip3 install llmcompressor
  fi

  if [ ! -d ${output_model_path} ]; then
    python3 ${SCRIPT_DIR}/llmcompressor_quantize.py \
      --model_path ${model_path} \
      --quant_method ${quant_method} \
      --saved_path ${output_model_path}
  else
    echo "The quantized hf model already exits in ${output_model_path}"
  fi

else
  echo "Usage: $0 hf_model_path quant_method(fp8, awq-w4a16, awq-w4a8) quantized_hf_model_path device_id(0,1)"
  exit
fi
