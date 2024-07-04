#!/usr/bin/env bash
################################################################################
# @Copyright: All Rights Reserved.
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-15 10:47:20
# @Details  : benchmark script for LMDeploy, vLLM and TensorRT-LLM backend.
################################################################################
set -euxo pipefail

function start_service(){
  local backend=$1
  local tp=$2
  local container_name=$3

  if [ $backend = "lmdeploy" ]; then
    docker run -d --gpus all --env CUDA_VISIBLE_DEVICES=$device_id \
      --privileged --shm-size=10g --ipc=host \
      -v ${model_path}:${model_path}  \
      -p ${service_port}:${service_port} \
      --name=${container_name} ${IMAGE_TAG} \
      lmdeploy serve api_server \
        ${model_path} \
        --server-name ${service_name} \
        --server-port ${service_port} \
        --tp ${tp} \
        --cache-max-entry-count 0.9 \
        --session-len 4096

  elif [ $backend = "vllm" ]; then
    docker run -d --gpus all --env CUDA_VISIBLE_DEVICES=$device_id \
      --privileged --shm-size=10g --ipc=host \
      -v ${model_path}:${model_path}  \
      -p ${service_port}:${service_port} \
      --name=${container_name} ${IMAGE_TAG} \
      python3 -m vllm.entrypoints.openai.api_server \
        --host ${service_name} \
        --port ${service_port} \
        --model ${model_path} \
        --dtype auto \
        -tp ${tp} \
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

    docker run -d --gpus all --env CUDA_VISIBLE_DEVICES=$device_id \
      --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      -v ${abs_model_path}:/app/${triton_model_repo} \
      -p ${service_port}:${service_port} \
      --name=${container_name} ${IMAGE_TAG} \
      bash -c "python /app/scripts/launch_triton_server.py \
        --http_port ${service_port} \
        --model_repo ${triton_model_repo} \
        --world_size ${tp}"

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
  local backend=$1
  local out_len=$2
  local bs=$3
  local num_requests=$4
  local log_path=$5

  if [ $backend = "vllm" ] || [ $backend = "lmdeploy" ]; then
    endpoint="/v1/completions"
  elif [ $backend = "tensorrt-llm" ]; then
    endpoint="/v2/models/ensemble/generate_stream"
  else
    echo "backend only supports vllm, lmdeploy or tensorrt-llm"
    exit
  fi

  python3  benchmarks/benchmark_serving.py \
      --backend ${backend} \
      --host ${service_name} \
      --port ${service_port} \
      --endpoint ${endpoint} \
      --tokenizer_name_or_path ${model_path} \
      --dataset_name "sharegpt" `# if using custom dataset, please set None` \
      --dataset_path ${test_data} \
      --request_output_len ${out_len} \
      --batch_size ${bs} \
      --num_requests ${num_requests} \
      --top_k 3 \
      --top_p 0.95 \
      --temperature 1e-7 \
      --repetition_penalty 1.15 \
      --log_path ${log_path}
}


if [ $# != 4 ]; then
  echo "Usage: $0 model_path data_path sample_num device_id(0 or 0,1 or 2,3 or 0,1,2,3)"
 exit
fi

model_path=$1
test_data=$2
sample_num=$3
device_id=$4

gpu_num=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")

service_name=0.0.0.0
service_port=800

#BACKEND="lmdeploy"
#BACKEND="vllm"
BACKEND="tensorrt-llm"
if [ $BACKEND = "lmdeploy" ]; then
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/wanzhen/lmdpeloy:0.4.2-arch_808990
elif [ $BACKEND = "vllm" ]; then
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/wanzhen/vllm:0.5.0.post1-arch_70808990
elif [ $BACKEND = "tensorrt-llm" ]; then
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/wanzhen/tensorrt-llm:0.12.0.dev2024070200-arch_808990
fi

tp_size="1"
gen_len="256 512"
batch_size="1 2"

for tp in ${tp_size};
do
  LOG_PATH=${BACKEND}-perf-tp${tp}.log

  for out_len in ${gen_len};
  do
    # if you want to make benchmark tests after launching server in container,
    # please COMMENT the start_service and stop_service, then modify the
    # service_name or service_port if necessary.
    container_name=benchmark-${BACKEND}-tp${tp}-${out_len}
    start_service ${BACKEND} $tp ${container_name}

    for bs in ${batch_size};
    do
      profile ${BACKEND} ${out_len} ${bs} ${sample_num} ${LOG_PATH}
    done

    stop_service ${container_name}
  done
done
