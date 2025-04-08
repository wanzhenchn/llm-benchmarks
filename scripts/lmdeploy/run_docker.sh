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

IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/lmdpeloy:0.7.1-arch_808990

docker run -it --gpus all --privileged --shm-size=10g \
            --ipc=host --network=host \
            -v /:/data \
            -v ~/:/root/ \
            -p ${port}:${port} \
            --rm --name=ld-test-${port} ${IMAGE_TAG} /bin/bash
