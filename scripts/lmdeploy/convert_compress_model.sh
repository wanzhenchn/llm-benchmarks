#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2023-09-25 16:19:07
# @Details  : compress hf models with awq-w4a16, fp8 quantization
################################################################################

set -euxo pipefail

if [ $# != 4 ]; then
  echo "Usage: $0 model_path deploy_model_format(hf or turbomind, modelopt) precision(fp16, awq-w4a16, w8a8 or fp8) device_id"
  exit 0
fi

MODEL_PATH="${1}/"
deploy_model_format=$2
precision=$3
dev_id=$4

export CUDA_VISIBLE_DEVICES=${dev_id}

model_name_dir=$(basename ${MODEL_PATH%/})
quant_model_path=${model_name_dir}-${precision}-${deploy_model_format}

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")


function check_transformers_version() {
  local model_path=$1

  echo "Checking transformer version from ${model_path} ..."
  transformers_version=$(grep 'transformers_version' ${model_path}/config.json | sed 's/.*"transformers_version": "\(.*\)",/\1/')

  python3 -c "
import sys
try:
    import transformers
    from packaging import version
    current_version = transformers.__version__
    target_version = '${transformers_version}'
    if version.parse(current_version) < version.parse('${transformers_version}'):
        sys.exit(1)
    else:
        sys.exit(0)
except ImportError:
    sys.exit(1)  # transformers isn't installed
" || {
    echo "Installing transformers==${transformers_version}"
    python3 -m pip install --no-cache-dir transformers==${transformers_version}
  }
}

# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/turbomind/deploy/converter.py#L205-L236
function convert_turbomind(){
  local model_type=$1  # llama/llama2/llama3/vicuna/qwen/internlm/baichuan2
  local model_format=$2
  local model_path=$3
  local group_size=$4
  local dst_path=$5

  lmdeploy convert \
    ${model_type} `# model_name` \
    ${model_path} `# model_path` \
    --model-format ${model_format} `# choose from ['llama', 'hf', 'awq', None]` \
    --group-size ${group_size} \
    --dst-path ${dst_path}

    # remove unuseful files
    rm -rf ${dst_path}/model_repository ${dst_path}/*.sh \
      ${dst_path}/triton_models/interactive \
      ${dst_path}/triton_models/postprocessing \
      ${dst_path}/triton_models/preprocessing
}


function quantize_model(){
  # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/cli/lite.py
  local method=$1 # calibrate, auto_awq, smooth_quant, auto_fp8
  local model_path=$2
  local dst_path=$3

  architecture=$(awk -F'["]' '/"architectures"/ {getline; print $2}' "${model_path}/config.json")
  intermediate_size=$(jq '.intermediate_size' "${model_path}/config.json")

  local extra_args=""

  if [ $method = calibrate ] || [ $method = smooth_quant ] || \
     [ $method = auto_awq ] || [ $method = auto_fp8 ]; then

       if [ $method = calibrate ] || [ $method = smooth_quant ]; then
         # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/lite/apis/calibrate.py#L113-L118
         # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/lite/apis/smooth_quant.py#L67-L72
         extra_args+="--calib-dataset wikitext2 "

      elif [ $method = auto_awq ]; then
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/lite/apis/auto_awq.py
        extra_args+="--calib-dataset wikitext2 "
        extra_args+="--w-bits 4 "
        extra_args+="--w-group-size 128 "

      elif [ $method = auto_fp8 ]; then
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/lite/apis/auto_fp8.py
        extra_args+="--calib-dataset ultrachat_2k "
        extra_args+="--act-scheme static " # static or dynamic
        extra_args+="--ignored-layer-list lm_head "
        if [ $architecture = MixtralForCausalLM ]; then
          # for MoE models, skip quantization for *block_sparse_moe.gate layers
          extra_args+="re:.*block_sparse_moe.gate "
        elif [ $architecture = CompassMoeForCausalLM ]; then
          extra_args+="re:.*mlp.gate re:.*block_sparse_moe.gate "
        elif [ $architecture = Qwen2_5_VLForConditionalGeneration ]; then
          extra_args+="re:.*visual "
        fi
      fi

      if [ $method = auto_awq ] && [ $architecture = Qwen2ForCausalLM ] && [ $intermediate_size = 29568 ]; then
        # Only for Qwen2.5-72 awq, padding the weights
        python3 ${SCRIPT_DIR}/padding_qwen2.5-72b.py ${model_path} "${model_path}-padded"
        # autoawq
        if [[ ! $(pip list | grep "autoawq") ]]; then
          python3 -m pip install autoawq
        fi
        python3 ${SCRIPT_DIR}/autoawq_quantize.py "${model_path}-padded" ${dst_path}
      else
        lmdeploy lite $method \
          ${model_path} `# The name or path of the model to be loaded.` \
          --calib-samples 128 \
          --calib-seqlen 2048 \
          --work-dir ${dst_path} \
          ${extra_args}
      fi
  else
    echo "quantization method only supports calibrate, auto_awq, smooth_quant."
    exit 1
  fi
}


function modelopt_quantize_model() {
  local model_path=$1
  local dst_path=$2

  local extra_args=""

  if [ $precision = awq-w4a16 ] || [ $precision = w8a8 ] || [ $precision = fp8 ]; then
    if [ $precision = awq-w4a16 ]; then
      extra_args+="--qformat int4_awq "

    elif [ $precision = w8a8 ]; then
      extra_args+="--qformat int8_sq "

    elif [ $precision = fp8 ]; then
      extra_args+="--qformat fp8 "
    fi
    python3 ${SCRIPT_DIR}/modelopt_quantize.py \
      --model-dir ${model_path} \
      --output-dir ${dst_path} \
      --calib-size 512 \
      --dataset ultrachat_2k \
      ${extra_args}
  else
    echo "Precision only supports awq-w4a16, w8a8 or fp8"
    exit 1
  fi
}


if [ $deploy_model_format = turbomind ]; then
  if [ $precision = fp16 ]; then
    convert_turbomind llama hf ${MODEL_PATH} 0 ${quant_model_path}

  elif [ $precision = awq-w4a16 ]; then
    # get hf awq model with quantization parameters
    quantize_model auto_awq ${MODEL_PATH} ${quant_model_path}-hf

    # convert hf-awq-quant models to turbomind format
    convert_turbomind llama awq ${quant_model_path}-hf 128 ${quant_model_path}

  elif [ $precision = fp8 ]; then
    # get hf fp8 model with quantization parameters
    quantize_model auto_fp8 ${MODEL_PATH} ${quant_model_path}-hf

    # convert hf-fp8-quant models to turbomind format
    convert_turbomind llama fp8 ${quant_model_path}-hf 0 ${quant_model_path}
  else
    echo "Precision only supports fp16, awq-w4a16, fp8"
    exit 1
  fi
elif [ $deploy_model_format = hf ]; then
  if [ $precision = fp16 ]; then
    echo "LMDeploy supports loading HuggingFace models without model conversion."
    exit 0

  elif [ $precision = awq-w4a16 ] || [ $precision = w8a8 ] || [ $precision = fp8 ]; then
    check_transformers_version ${MODEL_PATH}

    if [ $precision = awq-w4a16 ]; then
      quantize_model auto_awq ${MODEL_PATH} ${quant_model_path}

    elif [ $precision = w8a8 ]; then
      quantize_model smooth_quant ${MODEL_PATH} ${quant_model_path}

    elif [ $precision = fp8 ]; then
      quantize_model auto_fp8 ${MODEL_PATH} ${quant_model_path}
    fi
  else
    echo "Precision only supports fp16, awq-w4a16 or fp8"
    exit 1
  fi
elif [ $deploy_model_format = modelopt ]; then
  if [[ ! $(pip list | grep "nvidia-modelopt") ]]; then
    pip install "nvidia-modelopt[all]~=0.25.0" --extra-index-url https://pypi.nvidia.com
  fi
  modelopt_quantize_model ${MODEL_PATH} ${quant_model_path}
else
  echo "Deploying model only supports turbomind, hf or modelopt format"
  exit 1
fi
