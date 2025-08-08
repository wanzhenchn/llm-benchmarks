#! /bin/bash

GPU_PID=$1

# 获取 GPU PID 对应的 cgroup 信息
CGROUP_FILE="/proc/$GPU_PID/cgroup"
CONTAINER_ID=$(grep 'docker' "$CGROUP_FILE" | awk -F '/' '{print $3}' | sed -E 's/^docker-([0-9a-f]+)\.scope$/\1/' | head -n 1)


# 获取容器名称、镜像名称和挂载目录
CONTAINER_INFO=$(docker inspect "$CONTAINER_ID" 2>/dev/null)
CONTAINER_NAME=$(echo "$CONTAINER_INFO" | jq -r '.[0].Name' | sed 's/^\///')
IMAGE_NAME=$(echo "$CONTAINER_INFO" | jq -r '.[0].Config.Image')
MOUNTS=$(echo "$CONTAINER_INFO" | jq -c '.[0].Mounts[]')
MOUNT_DIR=$(echo "$CONTAINER_INFO" | jq -r '.[0].GraphDriver.Data.UpperDir')

echo "Container ID: $CONTAINER_ID"
echo "Container Name: $CONTAINER_NAME"
echo "Image Name: $IMAGE_NAME"
echo "$MOUNTS" | while IFS= read -r mount; do
    SOURCE=$(echo "$mount" | jq -r '.Source')
    DESTINATION=$(echo "$mount" | jq -r '.Destination')
    echo "- Host: $SOURCE -> Container: $DESTINATION"
done
