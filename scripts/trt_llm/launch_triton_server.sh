#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-26 10:32:15
# @Details  : launch triton service
################################################################################
set -euxo pipefail

if [ $# != 3  ]; then
  echo "Usage: $0 triton_model_repo http_port gpu_device_id(0,1)"
  exit
fi

triton_model_repo=$1
http_port=$2
device_id=$3

export CUDA_VISIBLE_DEVICES=${device_id}

tp=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")
grpc_port=801
metrics_port=802

python /app/scripts/launch_triton_server.py \
  --http_port ${http_port} \
  --grpc_port ${grpc_port} \
  --metrics_port ${metrics_port} \
  --model_repo=${triton_model_repo} \
  --world_size ${tp} \
  --log-level 0 `# 0: disabled, 1: verbose messages, 2: output all verbose messages of level <= 2`
#  --log-file triton_log.txt

# using `pkill tritonserver` to stop service
