#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-07-03 11:09:23
# @Details  : run lm-evaluation-harness with vLLM
################################################################################

set -euxo pipefail

# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md

if [ $# = 3 ] || [ $# = 4 ]; then
  mode=$1
  model_path=$2
  device_id=${!#}

  tp_size=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")
  output_path=results-$(basename ${model_path%/})-${mode}

  export CUDA_VISIBLE_DEVICES=${device_id}

  if [ $mode = fp8-kv-fp8 ]; then
    if [ $# = 4 ]; then
      quant_param_path=$3
    else
      echo "Usage1: $0 fp8-kv-fp8 model_path quantization_param_path device_id(like 0 or 0,1)"
      exit
    fi
  fi

  task_list="commonsense_qa,hellaswag,openbookqa,piqa,social_iqa,gsm8k,mmlu,boolq,scrolls_quality,ceval-valid"

  model_args="pretrained=${model_path},tensor_parallel_size=${tp_size},dtype=auto,gpu_memory_utilization=0.9,add_bos_token=True"

  if [ $mode = fp16 ]; then
    :
  elif [ $mode = fp8-kv-fp16 ]; then
    model_args+=",quantization=fp8"
  elif [ $mode = fp8-kv-fp8 ]; then
    model_args+=",quantization=fp8,kv_cache_dtype=fp8,quantization_param_path=${quant_param_path}"
  fi

  if [ ${tp_size} > 1 ]; then
    model_args+=",distributed_executor_backend=ray"
  fi

  lm_eval --model vllm \
    --model_args ${model_args} \
    --tasks ${task_list} \
    --trust_remote_code \
    --use_cache ${output_path}/${task_list} \
    --batch_size auto \
    --output_path ${output_path}

else
  echo "Usage1: $0 mode(fp16, fp8-kv-fp16) model_path device_id(like 0 or 0,1)"
  echo "Usage2: $0 mode(fp8-kv-fp8) model_path quantization_param_path device_id(like 0 or 0,1)"
fi
