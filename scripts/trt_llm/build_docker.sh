#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-22 14:49:32
# @Details  : Install TensorRt-LLM from source code or pip
################################################################################

#!/bin/bash

set -euxo pipefail

if [ $# != 1 ]; then
  echo "Usage: $0 install_type(ngc, all, trtllm_src, trtllm_pip)\n
                       all: install tritonserver, tensorrtllm-backend and tensorrt-llm from src \n
                       trtllm_src: install tensorrt-llm only from src \n
                       trtllm_pip: install tensorrt-llm only from pypi "
  exit
fi


if [ $1 = all ] || [ $1 = trtllm_src ]; then
  if [ $1 = all ] && [ ! -d tensorrtllm_backend ]; then
    git clone --recurse-submodules https://github.com/wanzhenchn/tensorrtllm_backend.git
  elif [ $1 = trtllm_src ] && [ ! -d TensorRT-LLM ]; then
    git clone --recurse-submodules https://github.com/wanzhenchn/TensorRT-LLM.git
  fi

  if [ $1 = all ]; then
    cd tensorrtllm_backend && git submodule update --init --recursive && git checkout dev
    TRT_LLM_VERSION=$(grep '^__version__' tensorrt_llm/tensorrt_llm/version.py | grep -o '=.*' | tr -d '= "')-triton
  elif [ $1 = trtllm_src ]; then
    cd TensorRT-LLM && git submodule update --init --recursive && git checkout dev
    TRT_LLM_VERSION=$(grep '^__version__' tensorrt_llm/version.py | grep -o '=.*' | tr -d '= "')
  fi

  CUDA_ARCHS="80-real;90-real"
  GIT_COMMIT=$(git rev-parse HEAD)
  BUILD_WHEEL_ARGS="--clean --trt_root /usr/local/tensorrt --python_bindings --benchmarks --cuda_architectures ${CUDA_ARCHS}"
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/tensorrt-llm:${TRT_LLM_VERSION}-arch_${CUDA_ARCHS//[^0-9]/}
else
  TRT_LLM_VERSION=25.01-trtllm-python-py3
  IMAGE_TAG=registry.cn-beijing.aliyuncs.com/devel-img/tensorrt-llm:${TRT_LLM_VERSION}

fi

if [ $1 = ngc ]; then
  DOCKER_BUILDKIT=1 docker build \
           --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver \
           --build-arg BASE_TAG=${TRT_LLM_VERSION} \
           -t ${IMAGE_TAG} \
           -f docker/Dockerfile.trt_llm_from_ngc .

elif [ $1 = all ]; then
  DOCKER_BUILDKIT=1 docker build \
           --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver \
           --build-arg BASE_TAG=25.01-py3 \
           --build-arg BUILD_WHEEL_ARGS="${BUILD_WHEEL_ARGS}" \
           --build-arg TORCH_INSTALL_TYPE="pypi" \
           --build-arg TRT_LLM_VER="${TRT_LLM_VERSION}" \
           --build-arg GIT_COMMIT="${GIT_COMMIT}" \
           -t ${IMAGE_TAG} \
           -f ../docker/Dockerfile.trt_llm_backend .

elif [ $1 = trtllm_src ]; then
  DOCKER_BUILDKIT=1 docker build \
           --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch \
           --build-arg BASE_TAG=25.01-py3 \
           --build-arg PYTHON_VERSION="3.12.3" \
           --build-arg BUILD_WHEEL_ARGS="${BUILD_WHEEL_ARGS}" \
           --build-arg TORCH_INSTALL_TYPE="skip" \
           --build-arg TRT_LLM_VER="${TRT_LLM_VERSION}" \
           --build-arg GIT_COMMIT="${GIT_COMMIT}" \
           --target release \
           -t ${IMAGE_TAG} \
           -f ../docker/Dockerfile.trt_llm_from_src .

elif [ $1 = trtllm_pip ]; then
  docker build -t ${IMAGE_TAG} -f docker/Dockerfile.trt_llm_from_pip .

else
  echo "Installation type only supports ngc, all, trtllm_src or trtllm_pip"
  exit
fi

docker push $IMAGE_TAG
