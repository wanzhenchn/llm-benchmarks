#!/usr/bin/env bash
################################################################################
# @Copyright: All Rights Reserved.
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-15 10:47:20
# @Details  : benchmark script for LMDeploy, vLLM and TensorRT-LLM backend.
################################################################################
set -euxo pipefail


function start_dcgm(){
  local dcgm_path=benchmarks/tools/dcgm
  while true; do
    if curl --fail --silent --output /dev/null http://localhost:9400/metrics; then
      echo "DCGM Service Ready."
      return 0
    else
      echo "DCGM Service not ready, starting first..."
      docker build -t dcgm:1.0 -f ${dcgm_path}/Dockerfile.dcgm ${dcgm_path}
      if docker ps -a --format '{{.Names}}' | grep -q "benchmark-dcgm"; then
        docker rm -f benchmark-dcgm
      fi
      docker run -d --name benchmark-dcgm --gpus all --cap-add SYS_ADMIN \
        --env DCGM_EXPORTER_COLLECTORS=/etc/dcgm-exporter/dcp-metrics.csv \
        -p 9400:9400 dcgm:1.0
      sleep 30
    fi
  done
}


function start_service(){
  local backend=$1
  local image_tag=$2
  local container_name=$3
  local model_path=$4
  local service_addr=$5
  local service_port=$6

  if [ $backend = "lmdeploy" ]; then
    docker run -d --gpus all --env NVIDIA_VISIBLE_DEVICES=$device_id \
      --privileged --shm-size=10g --ipc=host \
      -v ${model_path}:${model_path}  \
      -p ${service_port}:${service_port} \
      --name=${container_name} ${image_tag} \
      lmdeploy serve api_server \
        ${model_path} \
        --server-name ${service_addr} \
        --server-port ${service_port} \
        --tp ${gpu_num} \
        --cache-max-entry-count 0.9 \
        --session-len 4096 \
        --max-batch-size 256

  elif [ $backend = "vllm" ]; then
    docker run -d --gpus all --env NVIDIA_VISIBLE_DEVICES=$device_id \
      --privileged --shm-size=10g --ipc=host \
      -v ${model_path}:${model_path}  \
      -p ${service_port}:${service_port} \
      --name=${container_name} ${image_tag} \
      python3 -m vllm.entrypoints.openai.api_server \
        --host ${service_addr} \
        --port ${service_port} \
        --model ${model_path} \
        --dtype auto \
        -tp ${gpu_num} \
        --max-model-len 4096 \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.9 \
        --enable-prefix-caching \
        --swap-space 16 \
        --disable-log-stats \
        --disable-log-requests
#        --enforce-eager

  elif [ $backend = "tensorrt-llm" ]; then
    abs_model_path=$(realpath $model_path)
    triton_model_repo=$(basename ${abs_model_path%/})

    docker run -d --gpus all --env NVIDIA_VISIBLE_DEVICES=$device_id \
      --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      -v ${abs_model_path}:/app/${triton_model_repo} \
      -p ${service_port}:${service_port} \
      --name=${container_name} ${image_tag} \
      bash -c "python /app/scripts/launch_triton_server.py \
        --http_port ${service_port} \
        --model_repo ${triton_model_repo} \
        --world_size ${gpu_num}"

  else
    echo "backend only supports vllm, lmdeploy or tensorrt-llm"
    exit
  fi

  # wait 2m minute for service ready
  sleep 2m
}


function stop_service(){
  local container_name=$1

  container_id=$(docker ps -a | grep ${container_name} | awk '{print $1 }')
  docker stop ${container_id}
  docker rm ${container_id}
}


function profile()
{
  local host=$1
  local port=$2
  local log_path=$3
  local bs=$4
  local req_token_len=$5
  local monitor_device_ids=$6
  local get_gpu_metrics=$7
  local model_type=${8:-llm}
  local request_rate=${9:-"inf"}  # 默认无限速率，可以传入具体值如10.0; Default infinite rate, can be passed with specific value like 10.0
  local use_slo=${10:-"0"}  # 是否采用 SLO 方式进行压测
  # RandomDataset专用参数（可选，有默认值）; RandomDataset dedicated parameters (optional, with default values)
  local input_len=${11:-""}        # 空值表示使用原始数据集; An empty value indicates the use of the original dataset
  local range_ratio=${12:-0.0}    # RandomDataset输入长度变化比例; Length change ratio for RandomDataset input
  local num_requests=${13:-1000}  # RandomDataset请求数量; Number of requests for RandomDataset, Default 1000
  local output_range_ratio=${14:-0.0}  # RandomDataset输出长度变化比例; Length change ratio for RandomDataset output

  ## extra_body: pass other key:value pairs to the request body (JSON string; use true/false not True/False).
  # - for reasoning model: '{"reasoning_effort": "low", "include_reasoning": false}'
  # - example: '{"ignore_eos": true, "chat_template_kwargs": {"enable_thinking": false}}'

  ## dataset_name: if using custom dataset, please set None, default random

  # 公共参数设置; Common parameters setup
  local common_args=""
  local extra_args=""
  common_args+="--host ${host} "
  common_args+="--port ${port} "
  common_args+="--request_output_len ${req_token_len} "
  common_args+="--model_type ${model_type} "
  common_args+="--batch_size ${bs} "
  common_args+="--burstiness 1.0 "
  common_args+="--top_k 3 "
  common_args+="--top_p 0.95 "
  common_args+="--temperature 0.01 "
  common_args+="--repetition_penalty 1.15 "
  common_args+="--log_path ${log_path} "
  common_args+="--enable_expand_dataset true "
  common_args+="--get_gpu_metrics ${get_gpu_metrics} "
  common_args+="--get_gpu_metrics_freq 5 "
  common_args+="--device_ids ${monitor_device_ids} "
  common_args+="--get_llm_metrics false "

  # 根据input_len是否为空判断使用哪种模式; Determine which mode to use based on whether input_len is empty
  if [ -n "$input_len" ]; then
    # RandomDataset模式特有参数; RandomDataset mode specific parameters
    extra_args+="--dataset_name random "
    extra_args+="--dataset_path dummy "
    extra_args+="--request_rate ${request_rate} "
    local extra_body_json='{"ignore_eos":True,"chat_template_kwargs":{"enable_thinking":False}}'
    extra_args+="--extra_body ${extra_body_json} "
    extra_args+="--model_name_or_path ${model_path} "
    extra_args+="--num_requests ${num_requests} "
    extra_args+="--input_len ${input_len} "
    extra_args+="--range_ratio ${range_ratio} "
    extra_args+="--output_range_ratio ${output_range_ratio} "
    extra_args+="--prefix_len 0 "
    extra_args+="--seed 42 "
    extra_args+="--vocab_limit 100000 "
  else
    # 原始数据集模式特有参数; Original dataset mode specific parameters
    extra_args+="--dataset_name None "
    extra_args+="--dataset_path ${dataset_path} "
    extra_args+="--extra_body None "
  fi

  python3  ${SCRIPT_DIR}/benchmarks/benchmark_serving.py ${common_args} ${extra_args}
      
}


if [ $# != 7 ]; then
  echo "Usage: $0 backend(lmdeploy/vllm/sglang/tensorrt-llm) model_path model_type(llm/vlm) dataset_path port device_id(0 or 0,1) log_suffix"
  exit 1
fi

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

BACKEND=$1
model_path=$2
model_type=$3
dataset_path=$4
service_port=$5
device_id=$6
log_suffix=$7

if [ "$device_id" = all ]; then
  gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
  gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")
fi

DATASET="${dataset_path##*/}"
model_path="${model_path%/}"
MODEL="${model_path##*/}"

service_addr=0.0.0.0

if [ $BACKEND = "lmdeploy" ]; then
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/lmdpeloy:0.6.2-arch_808990
elif [ $BACKEND = "vllm" ]; then
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/vllm:0.6.3.post2.dev59-6c5af09b-arch_708090
elif [ $BACKEND = "tensorrt-llm" ]; then
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/tensorrt-llm:0.17.0.dev2024121700-arch_8090
fi

tp_size="2"
input_lens="4096 "
gen_lens="1024 "
batch_size="32 "

for tp in ${tp_size}; do
  # For local test, start dcgm to capture the GPU metrics, otherwise set get_gpu_metrics=false
  get_gpu_metrics=false
  if [ $get_gpu_metrics = true ]; then
    start_dcgm
  fi

  # if you want to make benchmark tests after launching server in container,
  # please COMMENT the start_service and stop_service, then modify the
  # service_addr or service_port if necessary.
  container_name=benchmark-${BACKEND}-tp${tp}
  start_service ${BACKEND} ${IMAGE_TAG} ${container_name} ${model_path} ${service_addr} ${service_port}

  LOG_PREFIX=perf-${BACKEND}-${MODEL}-${DATASET}
  REQUEST_RATE=${REQUEST_RATE:-"inf"}  # 默认无限速率; Default infinite rate

  if [ $DATASET = "random" ]; then
    echo "=== Run benchmark with random dataset ===="

    NUM_REQUESTS=${NUM_REQUESTS:-"50"}

    for input_len in ${input_lens}; do
      for out_len in ${gen_lens}; do
        LOG_PATH=${LOG_PREFIX}-${input_len}-${out_len}-tp${tp}-${log_suffix}.log

        for bs in ${batch_size}; do
          profile ${service_addr} ${service_port} ${LOG_PATH} ${bs} ${out_len} ${device_id} ${get_gpu_metrics} ${model_type} ${REQUEST_RATE} 0 ${input_len} 0 ${NUM_REQUESTS} 0
        done
      done
    done
  else
    echo "=== Run benchmark with business dataset ===="

    LOG_PATH=${LOG_PREFIX}-tp${tp}-${log_suffix}.log
    for out_len in ${gen_lens}; do
      for bs in ${batch_size}; do
        profile ${service_addr} ${service_port} ${LOG_PATH} ${bs} ${out_len} ${device_id} ${get_gpu_metrics} ${model_type} ${REQUEST_RATE}
      done
    done
  fi
  stop_service ${container_name}
done
