#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2023-09-25 14:49:32
# @Details  :
################################################################################
set -euxo pipefail


if [ $# = 1 ]; then
  # # Limit the number of parallel jobs to avoid OOM in ci
  max_jobs=$1
else
  max_jobs=$(nproc)
fi

if [ ! -d vllm ]; then
  git clone https://github.com/vllm-project/vllm
fi
cd vllm

if [ "$(pip list | grep setuptools-scm | wc -l)" -eq "0" ]; then
  python3 -m pip install setuptools-scm
fi

VERSION=$(python3 -c "from setuptools_scm import get_version; print(get_version())")
VERSION=$(echo "$VERSION" | sed 's/\.d[0-9]\{8\}//' | sed 's/+\(g\)\?/-/')
cuda_arch_list="9.0 10.0"
arch=arch_$(echo $cuda_arch_list | tr -d -c 0-9)

IMAGE_ADDR=registry.cn-beijing.aliyuncs.com/devel-img/vllm
IMAGE_TAG=${VERSION}-${arch}

docker build --pull \
  --build-arg CUDA_VERSION=12.8.1 --build-arg torch_cuda_arch_list="$cuda_arch_list" \
  --build-arg max_jobs=$max_jobs --build-arg nvcc_threads=1 \
  --build-arg RUN_WHEEL_CHECK=false \
  -t ${IMAGE_ADDR}:${IMAGE_TAG} \
  --target vllm-base -f docker/Dockerfile .
#  -f ../docker/Dockerfile.vllm .

docker push ${IMAGE_ADDR}:$IMAGE_TAG
