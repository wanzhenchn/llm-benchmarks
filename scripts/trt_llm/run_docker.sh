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

IMAGE_TAG=nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc0
# IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/tensorrt-llm:1.0.0rc5-arch_8090

docker run -it --gpus all --privileged \
  --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/:/root/ \
  -v /:/data \
  -p $port:$port \
  --entrypoint /bin/bash --rm --name=trt-test-$port ${IMAGE_TAG}
