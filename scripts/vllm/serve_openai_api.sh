#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-02 17:09:23
# @Details  : Start OpenAI compatible server
################################################################################

set -euxo pipefail

# https://docs.vllm.ai/en/latest/models/engine_args.html
# https://github.com/vllm-project/vllm/issues/3561#issuecomment-2019155747
# https://github.com/vllm-project/vllm/issues/3395#issuecomment-1997081715

host_name=0.0.0.0

if [ $# = 4 ] || [ $# = 5 ]; then
  mode=$1
  model_path=$2
  device_id=${!#}

  if [ $# = 5 ]; then
    module_path=$3
    port=$4
  else
    port=$3
  fi
  gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")
  export CUDA_VISIBLE_DEVICES=$device_id

  extra_args=""

  if [ $mode = fp16 ]; then
    extra_args+="--dtype auto " # “auto” will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models
    extra_args+="--enable-prefix-caching "

  elif [ $mode = w4a16 ]; then
    extra_args+="--dtype half "
    extra_args+="--enable-prefix-caching "
    extra_args+="--quantization awq"

  elif [ $mode = fp8 ]; then
    extra_args+="--dtype auto "
    extra_args+="--quantization fp8 "
    extra_args+="--kv-cache-dtype fp8 " # auto, fp8, fp8_e5m2, fp8_e4m3
    extra_args+="--quantization-param-path ${module_path} "

  elif [ $mode = lora ]; then
    extra_args+="--dtype auto "
    extra_args+="--enable-prefix-caching "
    extra_args+="--enable-lora "
    extra_args+="--max-lora-rank 32 "
    extra_args+="--lora-modules ${module_path}" #{name}={path} {name}={path}
  fi

  python3 -m vllm.entrypoints.openai.api_server \
    --host ${host_name} \
    --port $port \
    --model ${model_path} \
    -tp ${gpu_num} \
    --max-model-len 4096 \
    --max-num-seqs 256 `# How many requests can be batched into a single model run` \
    --gpu-memory-utilization 0.9 \
    --swap-space 16 `# CPU swap space size (GiB) per GPU.` \
    --disable-log-stats \
    --disable-log-requests \
    ${extra_args}
#    --uvicorn-log-level info `# ['debug', 'info', 'warning', 'error', 'critical', 'trace']`
#    --quantization ``
#    --enforce-eager `# By default it's not eager mode, If it's not set, it uses cuda_graph`
else
  echo "Usage1: $0 mode(fp16, w4a16 or kv8) model_path port device_id(0 or 0,1 or 1,3)"
  echo "Usage2: $0 mode(fp8) model_path kv_cache_scales_path port device_id(0 or 0,1 or 1,3)"
  echo "Usage3: $0 mode(lora) model_path lora_modules_path port device_id(0 or 0,1 or 1,3)"
  exit
fi
