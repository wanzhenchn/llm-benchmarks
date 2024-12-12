#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2023-09-25 16:19:07
# @Details  : Convert hf model to turbomind format or quantize/compress hf models
################################################################################

set -euxo pipefail

if [ $# != 4 ]; then
  echo "Usage: $0 model_path deploy_model_format(hf or turbomind) precision(fp16, w4a16, w8a8 or fp8) device_id"
  exit
fi

MODEL_PATH="${1}/"
deploy_model_format=$2
precision=$3
dev_id=$4

export CUDA_VISIBLE_DEVICES=${dev_id}

model_name_dir=$(basename ${MODEL_PATH%/})

turbomind_model_path=${model_name_dir}-turbomind
quant_turbomind_model_path=${model_name_dir}-${precision}-turbomind
quant_hf_model_path=${model_name_dir}-${precision}-hf


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

  extra_args=""

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
        if [ $architecture = MixtralForCausalLM ]; then
          # for MoE models, skip quantization for *block_sparse_moe.gate layers
          extra_args+="--ignored-layer-list lm_head re:.*block_sparse_moe.gate "
        fi
       fi

      lmdeploy lite $method \
        ${model_path} `# The name or path of the model to be loaded.` \
        --calib-samples 128 \
        --calib-seqlen 2048 \
        --work-dir ${dst_path} \
        ${extra_args}
  else
    echo "quantization method only supports calibrate, auto_awq, smooth_quant."
    exit
  fi
}


if [ $deploy_model_format = turbomind ]; then
  if [ $precision = fp16 ]; then
    convert_turbomind llama hf ${MODEL_PATH} 0 ${turbomind_model_path}

  elif [ $precision = w4a16 ]; then
    # get hf awq model with quantization parameters
    quantize_model auto_awq ${MODEL_PATH} ${quant_hf_model_path}

    # convert hf-awq-quant models to turbomind format
    convert_turbomind llama awq ${quant_hf_model_path} 128 ${quant_turbomind_model_path}

  elif [ $precision = fp8 ]; then
    # get hf fp8 model with quantization parameters
    quantize_model auto_fp8 ${MODEL_PATH} ${quant_hf_model_path}

    # convert hf-fp8-quant models to turbomind format
    convert_turbomind llama fp8 ${quant_hf_model_path} 0 ${quant_turbomind_model_path}
  else
    echo "Precision only supports fp16, w4a16, fp8"
    exit
  fi
elif [ $deploy_model_format = hf ]; then
  if [ $precision = fp16 ]; then
    echo "LMDeploy supports loading HuggingFace models without model conversion."
    exit

  elif [ $precision = w4a16 ] || [ $precision = w8a8 ] || [ $precision = fp8 ]; then
    check_transformers_version ${MODEL_PATH}

    if [ $precision = w4a16 ]; then
      quantize_model auto_awq ${MODEL_PATH} ${quant_hf_model_path}

    elif [ $precision = w8a8 ]; then
      quantize_model smooth_quant ${MODEL_PATH} ${quant_hf_model_path}

    elif [ $precision = fp8 ]; then
      quantize_model auto_fp8 ${MODEL_PATH} ${quant_hf_model_path}
    fi
  else
    echo "Precision only supports fp16, w4a16 or fp8"
    exit
  fi
else
  echo "Deploying model only supports turbomind or hf format"
  exit
fi
