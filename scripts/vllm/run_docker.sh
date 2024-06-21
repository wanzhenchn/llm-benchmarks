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

IMAGE_TAG=registry.cn-beijing.aliyuncs.com/wanzhen/vllm:0.5.0.post1-arch_70808990

docker run -it --gpus all --privileged --shm-size=10g \
            --ipc=host \
            -v ${PWD}:/workspace \
            -v /data/:/data \
            -p ${port}:${port} \
            --rm --name=vllm-test-${port} ${IMAGE_TAG} /bin/bash
