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
  local model_type=$1
  local host=$2
  local port=$3
  local req_token_len=$4
  local bs=$5
  local dataset_name=$6
  local dataset_path=$7
  local top_k=$8
  local top_p=$9
  local temperature=${10}
  local repetition_penalty=${11}
  local log_path=${12}
  local monitor_device_ids=${13}
  local get_gpu_metrics=${14}

  python3  ${SCRIPT_DIR}/benchmarks/benchmark_serving.py \
      --model_type ${model_type} \
      --host ${host} \
      --port ${port} \
      --dataset_name ${dataset_name} `# if using custom dataset, please set None, default sharegpt` \
      --dataset_path ${dataset_path} \
      --request_output_len ${req_token_len} \
      --batch_size ${bs} \
      --top_k ${top_k} \
      --top_p ${top_p} \
      --temperature ${temperature} \
      --repetition_penalty ${repetition_penalty} \
      --log_path ${log_path} \
      --enable_expand_dataset true \
      --get_gpu_metrics ${get_gpu_metrics} \
      --get_gpu_metrics_freq 5 \
      --device_ids "${monitor_device_ids}"
}


if [ $# != 12 ]; then
  echo "Usage: $0 backend(lmdeploy/vllm/tensorrt-llm) model_path model_type(llm/vlm) dataset_path dataset_name port top_k top_p temperature repetition_penalty device_id(0 or 0,1) log_name"
 exit 1
fi

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

BACKEND=$1
model_path=$2
model_type=$3
dataset_path=$4
dataset_name=$5
port=$6
top_k=$7 # 3
top_p=$8 # 0.95
temperature=$9 # 0.01
repetition_penalty=${10} # 1.15
device_id=${11}
log_name=${12}

if [ "$device_id" = all ]; then
  gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
  gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")
fi

service_addr=0.0.0.0
service_port=${port}

if [ $BACKEND = "lmdeploy" ]; then
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/lmdpeloy:0.6.2-arch_808990
elif [ $BACKEND = "vllm" ]; then
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/vllm:0.6.3.post2.dev59-6c5af09b-arch_708090
elif [ $BACKEND = "tensorrt-llm" ]; then
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/tensorrt-llm:0.17.0.dev2024121700-arch_8090
fi

tp_size="1"
gen_len="256 512"
batch_size="1 2"

for tp in ${tp_size}; do
  LOG_PATH=${BACKEND}-perf-${log_name}-tp${gpu_num}.log

  for out_len in ${gen_len}; do
    # For local test, start dcgm to capture the GPU metrics, otherwise COMMENT
    # start_dcgm and set get_gpu_metrics=false
    start_dcgm
    get_gpu_metrics=true

    # if you want to make benchmark tests after launching server in container,
    # please COMMENT the start_service and stop_service, then modify the
    # service_addr or service_port if necessary.
    container_name=benchmark-${BACKEND}-tp${tp}-${out_len}
    start_service ${BACKEND} ${IMAGE_TAG} ${container_name} ${model_path} ${service_addr} ${service_port}

    for bs in ${batch_size}; do
      profile ${model_type} ${service_addr} ${service_port} ${out_len} ${bs} ${dataset_name} ${dataset_path} \
        ${top_k} ${top_p} ${temperature} ${repetition_penalty} ${LOG_PATH} ${device_id} ${get_gpu_metrics}
    done

    stop_service ${container_name}
  done
done
