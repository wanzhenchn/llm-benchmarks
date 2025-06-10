#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2023-09-25 14:49:56
# @Details  :
################################################################################

set -euxo pipefail

if [ $# != 1 ]; then
  echo "Usage: $0 port"
  exit
fi

port=$1

# IMAGE_TAG=vllm/vllm-openai:v0.8.3
IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/vllm:0.9.1rc2.dev12-9368cc90b-arch_8090100

docker run -it --gpus all --privileged --shm-size=10g \
            --ipc=host --network=host \
            -v ~/:/root/ \
            -v /:/data \
            -p ${port}:${port} \
            --entrypoint /bin/bash \
            --rm --name=vllm-test-${port} ${IMAGE_TAG}
