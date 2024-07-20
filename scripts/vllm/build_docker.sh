#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2023-09-25 14:49:32
# @Details  :
################################################################################
set -euxo pipefail

if [ ! -d vllm ]; then
  git clone https://github.com/vllm-project/vllm
fi
cd vllm

VERSION=$(grep '^__version__' ./vllm/version.py | grep -o '=.*' | tr -d '= "')
cuda_arch_list="7.0 8.0 8.9 9.0"
arch=arch_$(echo $cuda_arch_list | tr -d -c 0-9)

IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/vllm:${VERSION}-${arch}

docker build --pull \
  --build-arg torch_cuda_arch_list="$cuda_arch_list" \
  -t ${IMAGE_TAG} \
  -f ../docker/Dockerfile.vllm .

docker push $IMAGE_TAG
