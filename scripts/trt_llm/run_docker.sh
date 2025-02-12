#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-22 14:49:32
# @Details  : run docker container
################################################################################

set -euxo pipefail

if [ $# != 1 ]; then
  echo "Usage: $0 port"
  exit
fi

port=$1

# IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/tensorrt-llm:0.17.0.dev2024121700-arch_8090
IMAGE_TAG=nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3

docker run -it --gpus all --privileged \
  --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ${PWD}:/workspace \
  -v /:/data \
  -p $port:$port \
  --rm --name=trt-test-$port \
  ${IMAGE_TAG} /bin/bash
