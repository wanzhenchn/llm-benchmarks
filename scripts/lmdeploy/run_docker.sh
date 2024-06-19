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

IMAGE_TAG=registry.cn-beijing.aliyuncs.com/wanzhen/lmdpeloy:0.4.2-arch_808990

docker run -it --gpus all --privileged --shm-size=10g \
            -v ${PWD}:/workspace \
            -v /data/:/data \
            -p ${port}:${port} \
            --rm --name=lmdeploy-test-${port} ${IMAGE_TAG} /bin/bash

