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

if [ $# = 5 ]; then
  model_path=$1
  mode=$2
  port=$3
  device_id=$4
  engine_type=$5

  gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")
  export CUDA_VISIBLE_DEVICES=$device_id

  extra_args=""

  if [ $mode = fp16 ] || [ $mode = w4a16 ] || [ $mode = kv-fp8 ] || [ $mode = fp8-kv-fp16 ] || [ $mode = fp8-kv-fp8 ]; then
    if [ $mode = w4a16 ]; then
      extra_args+="--dtype half "
    fi

    if [ $mode = fp8-kv-fp8 ] || [ $mode = kv-fp8 ]; then
      extra_args+="--kv-cache-dtype fp8 "
    fi

  else
    echo "mode only support fp16, w4a16, kv-fp8, fp8-kv-fp16 and fp8-kv-fp8"
    exit
  fi


  if [ $engine_type = 0 ]; then
    extra_args+="--num-scheduler-steps 8 "
  else
    export VLLM_USE_V1=1
  fi

  export VLLM_WORKER_MULTIPROC_METHOD="spawn"
  python3 -m vllm.entrypoints.openai.api_server \
    --host ${host_name} \
    --port $port \
    --model ${model_path} \
    -tp ${gpu_num} \
    --max-model-len 4096 \
    --max-num-seqs 256 `# How many requests can be batched into a single model run` \
    --gpu-memory-utilization 0.9 \
    --swap-space 16 `# CPU swap space size (GiB) per GPU.` \
    --trust-remote-code \
    --disable-log-stats \
    --disable-log-requests \
    --no-enable-prefix-caching \
    ${extra_args}
#    --enforce-eager `# By default it's not eager mode, If it's not set, it uses cuda_graph`
else
  echo "Usage1: $0 model_path mode(fp16, w4a16 or fp8-kv-fp8, fp8-kv-fp16) port device_id(0 or 0,1) engine_type(0 or 1)"
  exit
fi
