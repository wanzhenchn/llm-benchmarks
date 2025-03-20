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

if [ "$(pip list | grep setuptools-scm | wc -l)" -eq "0" ]; then
  python3 -m pip install setuptools-scm
fi

VERSION=$(python3 -c "from setuptools_scm import get_version; print(get_version())")
VERSION=$(echo "$VERSION" | sed 's/\.d[0-9]\{8\}//' | sed 's/+\(g\)\?/-/')
cuda_arch_list="8.0 9.0"
arch=arch_$(echo $cuda_arch_list | tr -d -c 0-9)

IMAGE_ADDR=registry.cn-beijing.aliyuncs.com/devel-img/vllm
IMAGE_TAG=${VERSION}-${arch}

docker build --pull \
  --build-arg torch_cuda_arch_list="$cuda_arch_list" \
  -t ${IMAGE_ADDR}:${IMAGE_TAG} \
  --target vllm-base -f Dockerfile .
#  -f ../docker/Dockerfile.vllm .

docker push ${IMAGE_ADDR}:$IMAGE_TAG
