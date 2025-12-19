#!/bin/bash

# =============================================================================
# vLLM Disaggregated Serving Script - P2P NCCL XpYd Architecture
# =============================================================================
# This script demonstrates disaggregated prefill and decode serving using
# P2P NCCL communication. The architecture supports various XpYd configurations:
#
# - 1P3D: 1 Prefill server + 3 Decode servers (current default)
# - 3P1D: 3 Prefill servers + 1 Decode server
# - etc.
#
# Configuration can be customized via environment variables:
#   MODEL: Model to serve
#   PREFILL_GPUS: Comma-separated GPU IDs for prefill servers
#   DECODE_GPUS: Comma-separated GPU IDs for decode servers
#   TP: Tensor parallel size (number of GPUs per instance, default: 1)
#   PREFILL_PORTS: Comma-separated ports for prefill servers
#   DECODE_PORTS: Comma-separated ports for decode servers
#   PROXY_PORT: Proxy server port used to setup XpYd connection.
#   TIMEOUT_SECONDS: Server startup timeout
# =============================================================================

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
MODEL=compass_max_v3_thinking_dev20251027-fp8-hf
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-600}
PROXY_PORT=${PROXY_PORT:-30001}

# Default 3P1D configuration (3 Prefill + 1 Decode)
PREFILL_GPUS=${PREFILL_GPUS:-0,1,2,3,4,5}
DECODE_GPUS=${DECODE_GPUS:-6,7}
TP=${TP:-2}
PREFILL_PORTS=${PREFILL_PORTS:-20003,20005,20007}
DECODE_PORTS=${DECODE_PORTS:-20009}

echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Tensor Parallel Size (TP): $TP"
echo "  Proxy Port: $PROXY_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=("disagg_proxy_p2p_nccl_xpyd.py")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        echo "Example: export HF_TOKEN=your_token_here"
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    # Check if the number of GPUs are >=2 via nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    pkill -9 -f "disagg_proxy_p2p_nccl_xpyd.py"
    kill -- -$$            # negative PID  ==  "this whole process-group"
    wait                   # reap children so we don't leave zombies
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 1
  done
}

main() {
    check_required_files
    check_hf_token
    check_num_gpus
#    pip install --ignore-installed pandas datasets quart numpy==2.2.0
#    pip install transformers -U
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - prefill*.log: Prefill server logs"
    echo "  - decode*.log: Decode server logs"
    echo "  - proxy.log: Proxy server log"

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo ""
    echo "Starting proxy server on port $PROXY_PORT..."
    python3 disagg_proxy_p2p_nccl_xpyd.py &
    PIDS+=($!)

    # Parse GPU and port arrays
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    # Validate TP configuration
    PREFILL_GPU_COUNT=${#PREFILL_GPU_ARRAY[@]}
    DECODE_GPU_COUNT=${#DECODE_GPU_ARRAY[@]}

    if [ $((PREFILL_GPU_COUNT % TP)) -ne 0 ]; then
        echo "Error: Prefill GPU count ($PREFILL_GPU_COUNT) must be divisible by TP ($TP)"
        exit 1
    fi
    if [ $((DECODE_GPU_COUNT % TP)) -ne 0 ]; then
        echo "Error: Decode GPU count ($DECODE_GPU_COUNT) must be divisible by TP ($TP)"
        exit 1
    fi

    # Calculate number of instances
    NUM_PREFILL_INSTANCES=$((PREFILL_GPU_COUNT / TP))
    NUM_DECODE_INSTANCES=$((DECODE_GPU_COUNT / TP))

    # Validate port arrays have enough ports
    if [ ${#PREFILL_PORT_ARRAY[@]} -lt $NUM_PREFILL_INSTANCES ]; then
        echo "Error: Not enough prefill ports. Need $NUM_PREFILL_INSTANCES ports, got ${#PREFILL_PORT_ARRAY[@]}"
        exit 1
    fi
    if [ ${#DECODE_PORT_ARRAY[@]} -lt $NUM_DECODE_INSTANCES ]; then
        echo "Error: Not enough decode ports. Need $NUM_DECODE_INSTANCES ports, got ${#DECODE_PORT_ARRAY[@]}"
        exit 1
    fi

    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo ""
    echo "Starting $NUM_PREFILL_INSTANCES prefill server(s) with TP=$TP (using $PREFILL_GPU_COUNT GPUs)..."
    for i in $(seq 0 $((NUM_PREFILL_INSTANCES - 1))); do
        # Calculate GPU IDs for this instance
        local start_idx=$((i * TP))
        local gpu_ids=()
        for j in $(seq 0 $((TP - 1))); do
            gpu_ids+=(${PREFILL_GPU_ARRAY[$((start_idx + j))]})
        done
        local gpu_list=$(IFS=','; echo "${gpu_ids[*]}")

        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((21001 + i * TP))

        echo "  Prefill server $((i+1)): GPUs $gpu_list (TP=$TP), Port $port, KV Port $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_list vllm serve $MODEL \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size $TP \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --quantization fp8 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > prefill$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # =============================================================================
    # Launch Decode Servers (Y Decoders)
    # =============================================================================
    echo ""
    echo "Starting $NUM_DECODE_INSTANCES decode server(s) with TP=$TP (using $DECODE_GPU_COUNT GPUs)..."
    for i in $(seq 0 $((NUM_DECODE_INSTANCES - 1))); do
        # Calculate GPU IDs for this instance
        local start_idx=$((i * TP))
        local gpu_ids=()
        for j in $(seq 0 $((TP - 1))); do
            gpu_ids+=(${DECODE_GPU_ARRAY[$((start_idx + j))]})
        done
        local gpu_list=$(IFS=','; echo "${gpu_ids[*]}")

        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i * TP))

        echo "  Decode server $((i+1)): GPUs $gpu_list (TP=$TP), Port $port, KV Port $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_list vllm serve $MODEL \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size $TP \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --quantization fp8 \
        --trust-remote-code \
        --gpu-memory-utilization 0.8 \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > decode$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for i in $(seq 0 $((NUM_PREFILL_INSTANCES - 1))); do
        local port=${PREFILL_PORT_ARRAY[$i]}
        if ! wait_for_server $port; then
            echo "Failed to start prefill server on port $port"
            cleanup
            exit 1
        fi
    done
    for i in $(seq 0 $((NUM_DECODE_INSTANCES - 1))); do
        local port=${DECODE_PORT_ARRAY[$i]}
        if ! wait_for_server $port; then
            echo "Failed to start decode server on port $port"
            cleanup
            exit 1
        fi
    done

    echo ""
    echo "All servers are up. Starting benchmark..."

    # =============================================================================
    # Run Benchmark
    # =============================================================================

    # Calculate XpYd configuration (based on instances, not GPUs)
    NUM_PREFILL=${#PREFILL_GPU_ARRAY[@]}
    NUM_DECODE=${#DECODE_GPU_ARRAY[@]}
    CONFIG_NAME="${NUM_PREFILL_INSTANCES}P${NUM_DECODE_INSTANCES}D"

    ISL=8000
    OSL=100
    NUM_PROMPTS=500
    LABEL=perf-${MODEL##*/}-NUM${NUM_PROMPTS}-ISL${ISL}-OSL${OSL}-${CONFIG_NAME}
    PERF_LOG=${LABEL}.log
    PERF_FILENAME=${LABEL}.json


    # Output p/d configuration to benchmark.log
    {
        echo "============================================================================="
        echo "P/D Configuration Information"
        echo "============================================================================="
        echo "Configuration: $CONFIG_NAME (${NUM_PREFILL_INSTANCES} Prefill + ${NUM_DECODE_INSTANCES} Decode instances)"
        echo "Tensor Parallel Size (TP): $TP"
        echo "Model: $MODEL"
        echo "Prefill GPUs: $PREFILL_GPUS (${PREFILL_GPU_COUNT} GPUs, ${NUM_PREFILL_INSTANCES} instances)"
        echo "Prefill Ports: $PREFILL_PORTS"
        echo "Decode GPUs: $DECODE_GPUS (${DECODE_GPU_COUNT} GPUs, ${NUM_DECODE_INSTANCES} instances)"
        echo "Decode Ports: $DECODE_PORTS"
        echo "Proxy Port: $PROXY_PORT"
        echo "============================================================================="
        echo ""
    } | tee ${PERF_LOG}

    dataset_name=random # sharegpt # random

    for bs in 2 4 8 16 32; do
        extra_args=""
        if [ $dataset_name = sharegpt ]; then
          extra_args+="--dataset-name sharegpt "
          extra_args+="--dataset-path ShareGPT_V3_unfiltered_cleaned_split.json "
          extra_args+="--sharegpt-output-len ${OSL} "
        elif [ $dataset_name = random ]; then
#          extra_args+="--seed $(date +%s) "
          extra_args+="--random-input-len ${ISL} "
          extra_args+="--random-output-len ${OSL} "
        fi

        vllm bench serve --port 10001 \
            --model $MODEL \
            --num-prompts ${NUM_PROMPTS} \
            --max-concurrency ${bs} \
            --num-warmups 50 \
            --trust-remote-code \
            --append-result --result-filename ${PERF_FILENAME} ${extra_args} | tee -a ${PERF_LOG}
#            --request-rate ${req_rate} --burstiness 100 \
    done

    echo "Benchmarking done. Cleaning up..."

    cleanup
}

main
