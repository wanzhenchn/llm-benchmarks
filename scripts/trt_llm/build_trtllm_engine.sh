#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-22 18:47:18
# @Details  : convert hf model, build trtllm engines, and run perf summary.
################################################################################

set -euxo pipefail

if [ $# != 6  ]; then
  echo "Usage: $0 hf_model_path precision(fp16, int4, kv8, int4_kv8, w8a8, fp8) hf_model_converted_path engine_path tp_size device_id(0,1)"
  exit
fi

model_path=$1
precision=$2
converted_ckpt_path=$3
engine_path=$4
tp=$5
device_id=$6
# tp=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")

saved_ckpt_path=${converted_ckpt_path}-${tp}-${precision}

export CUDA_VISIBLE_DEVICES=${device_id}

# Credit to: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama

function convert_checkpoint {
  local precision=$1
  local output_path=$2
  local hf_model_path=$3
  local tp_size=$4

  extra_args=""
  if [ ! -d ${output_path} ]; then
    if [ $precision = fp16 ] || [ $precision = kv8 ] || [ $precision = w8a8 ]; then
      if [ $precision = kv8 ]; then
        extra_args+="--int8_kv_cache"
      elif [ $precision = w8a8 ]; then
        # SmoothQuant INT8
        extra_args+="--smoothquant 0.5 "
        extra_args+="--per_token "
        extra_args+="--per_channel "
      fi

      python /app/tensorrt_llm/examples/llama/convert_checkpoint.py \
        --model_dir ${hf_model_path} \
  	    --tp_size ${tp_size} \
  	    --dtype float16 \
  	    --output_dir ${output_path} \
        ${extra_args}

    elif [ $precision = int4 ] || [ $precision = int4_kv8 ] || [ $precision = fp8 ]; then
      if [ $precision = int4 ] || [ $precision = int4_kv8 ]; then
        extra_args+="--qformat int4_awq "
        extra_args+="--awq_block_size 128 "
        extra_args+="--calib_size 32 "
        if [ $precision = int4_kv8 ]; then
          extra_args+="--kv_cache_dtype int8"
        fi
      elif [ $precision = fp8 ]; then
        extra_args+="--qformat fp8 "
        extra_args+="--kv_cache_dtype fp8 "
        extra_args+="--calib_size 512"

        if [[ ! $(pip list | grep "nvidia-modelopt") ]]; then
          # NVIDIA Modelopt (AlgorithMic Model Optimization) toolkit for the model quantization process.
          pip install "nvidia-modelopt[all]~=0.11.0" --extra-index-url https://pypi.nvidia.com
        fi
      fi

      python /app/tensorrt_llm/examples/quantization/quantize.py \
        --model_dir ${hf_model_path} \
  	    --dtype float16 \
  	    --tp_size ${tp_size} \
  	    --output_dir ${output_path} \
        ${extra_args}

    fi
  else
    echo "The converted hf model already exits in ${output_path}"
  fi
}


function build_tensorrt_engine {
  local precision=$1
  local converted_checkpoint=$2
  local max_batch_size=$3
  local max_input_len=$4
  local max_output_len=$5
  local output_path=$6

  # https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md#maximum-number-of-tokens
  # max_batch_size * max_input_len * alpha + max_batch_size * max_beam_width * (1 - alpha)
  alpha=0.1
  max_beam_width=1
  max_num_tokens=$(echo "${max_batch_size} ${max_input_len} ${alpha} ${max_beam_width}" | awk '{print int($1*$2*$3+$1*$4*(1-$3))}')

  # If no nvlink, disable custom all reduce.
  extra_args=""
  if [ "$(nvidia-smi nvlink -s | wc -l)" -eq "0"  ] || [ $(nvidia-smi nvlink --status | grep inActive | wc -l) -ge 1 ]; then
    extra_args+="--use_custom_all_reduce=disable "
  fi

  if [ $precision = fp8 ]; then
    extra_args+="--use_fp8_context_fmha enable"
  fi

  trtllm-build --checkpoint_dir ${converted_checkpoint} \
    --max_batch_size ${max_batch_size} \
    --max_input_len ${max_input_len} \
    --max_output_len ${max_output_len} \
    --max_num_tokens ${max_num_tokens} \
    --gemm_plugin auto \
    --gpt_attention_plugin float16 \
    --remove_input_padding enable \
    --paged_kv_cache enable \
    --workers 4 \
    --output_dir ${output_path} \
    ${extra_args}
}


function run_perf_summary {
  local hf_model_path=$1
  local trt_engine_path=$2
  local tp_size=$3

  if [ $tp_size = 1 ]; then
    python /app/tensorrt_llm/examples/summarize.py \
      --test_trt_llm \
      --data_type fp16  \
      --hf_model_dir ${hf_model_path}  \
      --engine_dir ${trt_engine_path}
  else
     mpirun -n $tp_size --allow-run-as-root \
       python /app/tensorrt_llm/examples/summarize.py \
       --test_trt_llm \
       --data_type fp16  \
       --hf_model_dir ${hf_model_path}  \
       --engine_dir ${trt_engine_path}
  fi
}


function run_engine {
  local hf_model_path=$1
  local trt_engine_path=$2

  python /app/tensorrt_llm/examples/run.py \
    --engine_dir=${trt_engine_path} \
    --tokenizer_dir ${hf_model_path} \
    --input_text "What is machine learning?"
    --max_output_len 100 \
    --streaming \
    --top_k 3 \
    --top_p 0.95 \
    --temperature 1e-7 \
    --repetition_penalty 1.15
}


max_batch_size=64
max_input_len=2048
max_output_len=1024

convert_checkpoint ${precision} ${saved_ckpt_path} ${model_path} ${tp}
build_tensorrt_engine ${precision} ${saved_ckpt_path} ${max_batch_size} ${max_input_len} ${max_output_len} ${engine_path}
run_perf_summary ${model_path} ${engine_path} ${tp}
# run_engine ${model_path} ${engine_path}
