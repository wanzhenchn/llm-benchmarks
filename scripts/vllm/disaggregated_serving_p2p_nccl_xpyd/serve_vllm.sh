#!/bin/bash

# =============================================================================
# vLLM Serving Script
# =============================================================================
# Configuration can be customized via environment variables:
#   MODEL: Model to serve
#   GPUS: Comma-separated GPU IDs for servers
#   PORTS: ports for servers
#   TIMEOUT_SECONDS: Server startup timeout
# =============================================================================

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
MODEL=compass_max_v3_thinking_dev20251027
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-600}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-10002}

NUM_GPU=$(echo $GPUS | tr ',' '\n' | wc -l)

echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  GPUs: $GPUS, Port: $PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"


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
        echo "You need at least 2 GPUs to run baseline."
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
    if curl -s "localhost:${port}/v1/models" > /dev/null; then
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

    echo "Launching baseline serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - baseline*.log: baseline server logs"


    # =============================================================================
    # Launch Servers
    # =============================================================================
    echo ""
    echo "Starting baseline server(s)..."

    CUDA_VISIBLE_DEVICES=$GPUS vllm serve $MODEL \
    --host 0.0.0.0 \
    --port ${PORT} \
    --tensor-parallel-size ${NUM_GPU} \
    --seed 1024 \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --quantization fp8 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 > baseline.log 2>&1 &
    PIDS+=($!)


    # =============================================================================
    # Wait for Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for servers to start..."
    if ! wait_for_server $PORT; then
        echo "Failed to start server on port $PORT"
        cleanup
        exit 1
    fi

    echo ""
    echo "Baseline servers are up. Starting benchmark..."

    # =============================================================================
    # Run Benchmark
    # =============================================================================


    ISL=8000
    OSL=100
    NUM_PROMPTS=500
    LABEL=perf-${MODEL##*/}-NUM${NUM_PROMPTS}-ISL${ISL}-OSL${OSL}-TP${NUM_GPU}
    PERF_LOG=${LABEL}.log
    PERF_FILENAME=${LABEL}.json


    # Output p/d configuration to benchmark.log
    {
        echo "============================================================================="
        echo "Configuration Information"
        echo "============================================================================="
        echo "Configuration: TP=${NUM_GPU}"
        echo "Model: $MODEL"
        echo "GPUs: $GPUS"
        echo "Port: $PORT"
        echo "============================================================================="
        echo ""
    } | tee ${PERF_LOG}

    dataset_name=random #sharegpt # random

    for bs in 2 4 8 16 32 ; do
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

        vllm bench serve --port ${PORT} \
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
