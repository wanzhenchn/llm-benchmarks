#!/usr/bin/env bash

set -euxo pipefail

if [ $# != 1 ]; then
  echo "Usage: $0 port"
  exit
fi

port=$1

IMAGE_TAG=rocm/ali-private:ubuntu22.04_rocm7.2.0.43_cp310_torch2.9.1_sglang_4d921e5_aiter_6fe675a_qwen3_5_20260327_v2

docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_PTRACE \
           --security-opt seccomp=unconfined --group-add video \
           --privileged --shm-size=10g --ipc=host --network=host \
           -v ~/:/root/ \
           -v /mnt/raid0:/data \
           -p ${port}:${port} \
           --entrypoint /bin/bash \
           --rm --name=sgl-test-${port} ${IMAGE_TAG}
