#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2023-09-25 14:49:32
# @Details  : Install lmdeploy from source code
################################################################################


set -euxo pipefail


if [ ! -d lmdeploy ]; then
  git clone https://github.com/InternLM/lmdeploy.git
fi
cd lmdeploy

CUDA_ARCHS="80;89;90"
VERSION=$(grep '^__version__' ./lmdeploy/version.py | grep -o '=.*' | tr -d "= '")
IMAGE_TAG=registry.cn-beijing.aliyuncs.com/wanzhen/lmdpeloy:${VERSION}-arch_${CUDA_ARCHS//[^0-9]/}

docker build --progress auto \
  --build-arg SM=${CUDA_ARCHS} \
  -t ${IMAGE_TAG} \
  -f ../docker/Dockerfile.lmdeploy ..

docker push $IMAGE_TAG
