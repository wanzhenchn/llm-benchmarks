#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-22 18:47:18
# @Details  : convert hf model, build trtllm engines and run perf summary.
################################################################################

set -euxo pipefail

if [ $# != 6  ]; then
  echo "Usage: $0 hf_model_path model_type(llama/qwen/moe) precision(fp16, kv-int8, awq-w4a16, awq-w4a8, sq-w8a8, fp8, w4aINT8) engine_path tp_size device_id(0,1)"
  exit
fi

model_path=$1
model_type=$2
precision=$3
engine_path=$4
tp=$5
device_id=$6
# tp=$(echo "$device_id" |grep -o "[0-9]" |grep -c "")

converted_ckpt_path=$(basename "$model_path")-tp${tp}-${precision}

export CUDA_VISIBLE_DEVICES=${device_id}


function create_py_virtualenv {
  local VENV_PATH=$1

  if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
  else
    echo "No virtual environment found in $VENV_PATH. Creating one..."
    virtualenv -p `which python3` "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
  fi
}


# Credit to: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama
function convert_checkpoint {
  local precision=$1
  local output_path=$2
  local hf_model_path=$3
  local tp_size=$4

  extra_args=""
  if [ ! -d ${output_path} ]; then
    if [ $precision = fp16 ] || [ $precision = kv-int8 ] || [ $precision = sq-w8a8 ] || [ $precision = w4aINT8 ]; then
      if [ $precision = kv-int8 ]; then
        extra_args+="--int8_kv_cache"

      elif [ $precision = sq-w8a8 ]; then
        # SmoothQuant INT8
        extra_args+="--smoothquant 0.5 "
        extra_args+="--per_token "
        extra_args+="--per_channel "

      elif [ $precision = w4aINT8 ]; then
        qserve_ckpt_path=$(basename "$hf_model_path")-qserve-$precision

        if [ ! -d ${qserve_ckpt_path} ]; then
          if [ -z "${VIRTUAL_ENV:-}" ]; then
            PY3_ENV=/opt/py3_env
            create_py_virtualenv $PY3_ENV

            # QServe w4a8: https://github.com/mit-han-lab/deepcompressor/tree/main/examples/llm
            if [[ ! $(pip list | grep "deepcompressor") ]]; then
              git clone https://github.com/mit-han-lab/deepcompressor.git && cd deepcompressor
              # fix installation promblems
              sed -i '/image_reward = { git = "git@github.com:THUDM\/ImageReward.git", branch = "main" }/c\image_reward = { git = "https://github.com/THUDM/ImageReward.git", branch = "main" }' pyproject.toml
              pip install -e . && cd -
            fi

            python3 -m deepcompressor.app.llm.ptq deepcompressor/examples/llm/configs/qoq-gchn.yaml \
              --model-name $(basename "$hf_model_path") \
              --model-path  ${hf_model_path} \
              --smooth-proj-alpha 0 --smooth-proj-beta 1 \
              --smooth-attn-alpha 0.5 --smooth-attn-beta 0 \
              --save-model ${qserve_ckpt_path} \
              --eval-evaluators lm_eval \
              --eval-tasks gsm8k

            deactivate
          fi
        else
          echo "The converted qserve model already exits in ${qserve_ckpt_path}, skipped"
        fi

        extra_args+="--quant_ckpt_path ${qserve_ckpt_path} "
        extra_args+="--use_qserve "
#        extra_args+="--per_group " # Add this option if using per-group quantization
	      export TRTLLM_DISABLE_UNIFIED_CONVERTER=1  # The current checkpoint conversion code requires legacy path
      fi

      # https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/mixtral
      if [ $model_type = moe ]; then
        extra_args+="--moe_tp_size ${tp_size} "
#        extra_args+="--moe_ep_size ${ep_size} "
      fi

      python3 /app/examples/${model_type}/convert_checkpoint.py \
        --model_dir ${hf_model_path} \
  	    --tp_size ${tp_size} \
  	    --dtype float16 \
  	    --output_dir ${output_path} ${extra_args}

    elif [ $precision = awq-w4a16 ] || [ $precision = awq-w4a8 ] || [ $precision = fp8 ]; then
      if [ $precision = awq-w4a16 ] || [ $precision = awq-w4a8 ]; then
        if [ $precision = awq-w4a8 ]; then
          extra_args+="--qformat w4a8_awq "
        else
          extra_args+="--qformat int4_awq "
          extra_args+="--awq_block_size 128 "
        fi
        extra_args+="--calib_size 512 "
        extra_args+="--kv_cache_dtype int8"
      elif [ $precision = fp8 ]; then
        extra_args+="--qformat fp8 "
        extra_args+="--kv_cache_dtype fp8 "
        extra_args+="--calib_size 512"

        if [[ ! $(pip list | grep "nvidia-modelopt") ]]; then
          # NVIDIA Modelopt (AlgorithMic Model Optimization) toolkit for the model quantization process.
          pip install "nvidia-modelopt[all]~=0.23.0" --extra-index-url https://pypi.nvidia.com
        fi
      fi

      python3 /app/examples/quantization/quantize.py \
        --model_dir ${hf_model_path} \
  	    --dtype float16 \
  	    --tp_size ${tp_size} \
  	    --output_dir ${output_path} \
        ${extra_args}

    fi
  else
    echo "The converted hf model already exits in ${output_path}, skipped"
  fi
}


function build_tensorrt_engine {
  # https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md
  local precision=$1
  local converted_checkpoint=$2
  local hf_model_path=$3
  local max_batch_size=$4
  local max_seq_len=$5
  local max_num_tokens=$6
  local output_path=$7

  # If no nvlink, disable custom all reduce.
  extra_args=""
  if [ "$(nvidia-smi nvlink -s | wc -l)" -eq "0"  ] || [ $(nvidia-smi nvlink --status | grep inActive | wc -l) -ge 1 ]; then
    extra_args+="--use_custom_all_reduce=disable "
  else
    extra_args+="--reduce_fusion enable " # only supported for the llama model
  fi

  if [ $precision = fp8 ]; then
    extra_args+="--use_fp8_context_fmha enable "
#    extra_args+="--gemm_plugin fp8 " # only recommended for latency reduction in small-batch-size(<=4) scenarios
    extra_args+="--use_fused_mlp=enable "
    extra_args+="--gemm_swiglu_plugin fp8 "
#    extra_args+="--low_latency_gemm_swiglu_plugin fp8" # for small batch size
  else
    extra_args+="--gemm_plugin auto "
  fi

  trtllm-build --checkpoint_dir ${converted_checkpoint} \
    --max_batch_size ${max_batch_size} \
    --max_seq_len ${max_seq_len} \
    --max_num_tokens ${max_num_tokens} \
    --remove_input_padding enable \
    --kv_cache_type=paged
    --multiple_profiles enable \
    --workers 4 \
    --output_dir ${output_path} \
    ${extra_args}
#    --gpt_attention_plugin float16 \

  # copy tokenizer model to engine path for launching OpenAI-Compatible server
  cp ${hf_model_path}/tokenizer* ${output_path}
}


function run_perf_summary {
  local hf_model_path=$1
  local trt_engine_path=$2
  local tp_size=$3

  if [ $tp_size = 1 ]; then
    python3 /app/examples/summarize.py \
      --test_trt_llm \
      --data_type fp16  \
      --hf_model_dir ${hf_model_path}  \
      --engine_dir ${trt_engine_path}
  else
     mpirun -n $tp_size --allow-run-as-root \
       python3 /app/examples/summarize.py \
       --test_trt_llm \
       --data_type fp16  \
       --hf_model_dir ${hf_model_path}  \
       --engine_dir ${trt_engine_path}
  fi
}


function run_engine {
  local hf_model_path=$1
  local trt_engine_path=$2

  python3 /app/examples/run.py \
    --engine_dir=${trt_engine_path} \
    --tokenizer_dir ${hf_model_path} \
    --input_text "What is machine learning?"
    --max_output_len 50 \
    --streaming \
    --top_k 3 \
    --top_p 0.95 \
    --temperature 0.01 \
    --repetition_penalty 1.15
}


max_batch_size=256
max_input_len=2048
max_output_len=1024
max_seq_len=$(echo "${max_input_len} ${max_output_len}" | awk '{print int($1+$2)}')
max_num_tokens=8192

convert_checkpoint ${precision} ${converted_ckpt_path} ${model_path} ${tp}
build_tensorrt_engine ${precision} ${converted_ckpt_path} ${model_path} ${max_batch_size} ${max_seq_len} ${max_num_tokens} ${engine_path}
run_perf_summary ${model_path} ${engine_path} ${tp}
# run_engine ${model_path} ${engine_path}
