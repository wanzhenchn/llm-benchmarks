#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-06-28 14:49:32
# @Details  : Install llm-evaluation-harness
################################################################################


set -euxo pipefail

if [ $# != 1 ]; then
  echo "Usage: $0 backend(lmdeploy, vllm or ttr-llm)"
  exit
fi

BACKEND=$1

if [ ! -d llm-evaluation-harness ]; then
  git clone https://github.com/EleutherAI/lm-evaluation-harness
fi

if [ $BACKEND = "lmdeploy" ]; then
  BASE_IMAGE=registry.cn-beijing.aliyuncs.com/wanzhen/lmdpeloy
  VERSION=0.4.2-arch_808990
  backend=lmdeploy
elif [ $BACKEND = "vllm" ]; then
  BASE_IMAGE=registry.cn-beijing.aliyuncs.com/wanzhen/vllm
  VERSION=0.5.0.post1-arch_808990
  backend=vllm
elif [ $BACKEND = "trt-llm" ]; then
  BASE_IMAGE=registry.cn-beijing.aliyuncs.com/wanzhen/tensorrt-llm
  VERSION=0.11.0.dev2024062500-arch_808990
  backend=tensorrt-llm
fi

IMAGE_TAG=${BASE_IMAGE}:${VERSION}-harness

docker build --progress auto \
  --build-arg BASE_IMAGE=${BASE_IMAGE} \
  --build-arg VERSION=${VERSION} \
  --build-arg BACKEND=${backend} \
  -t ${IMAGE_TAG} \
  -f docker/Dockerfile.evaluation_harness .

docker push $IMAGE_TAG
