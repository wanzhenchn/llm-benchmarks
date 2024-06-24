#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2023-09-25 16:19:07
# @Details  : Convert hf model to turbomind format or quantize/compress hf models
################################################################################

set -euxo pipefail

if [ $# != 4 ]; then
  echo "Usage: $0 model_path deploy_model_format(hf or turbomind) precision(fp16, w4a16 or kv8) device_id"
  exit
fi

MODEL_PATH=$1
deploy_model_format=$2
precision=$3
dev_id=$4

export CUDA_VISIBLE_DEVICES=${dev_id}

model_name_dir=$(basename ${MODEL_PATH%/})

turbomind_model_path=${model_name_dir}-turbomind
quant_turbomind_model_path=${model_name_dir}-${precision}-turbomind
quant_hf_model_path=${model_name_dir}-${precision}-hf

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
  local method=$1 # calibrate, auto_awq, smooth_quant
  local model_path=$2
  local dst_path=$3

  if [ $method = calibrate ] || [ $method = smooth_quant ]; then
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/lite/apis/calibrate.py#L113-L118
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/lite/apis/smooth_quant.py#L67-L72
    lmdeploy lite $method \
      ${model_path} `# The name or path of the model to be loaded.` \
      --calib-dataset 'wikitext2' \
      --calib-samples 128 \
      --calib-seqlen 2048 \
      --work-dir ${dst_path}

  elif [ $method = auto_awq ]; then
    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/lite/apis/auto_awq.py#L32-L40
    lmdeploy lite auto_awq \
      ${model_path} `# The name or path of the model to be loaded.` \
      --calib-dataset 'wikitext2' \
      --calib-samples 128 \
      --calib-seqlen 2048 \
      --w-bits 4 \
      --w-group-size 128 \
      --work-dir ${dst_path}
  else
    echo "quantization method only supports calibrate, auto_awq, smooth_quant."
    exit
  fi
}


if [ $deploy_model_format = turbomind ]; then
  if [ $precision = fp16 ]; then
    convert_turbomind llama hf ${MODEL_PATH} 0 ${turbomind_model_path}

  elif [ $precision = w4a16 ]; then
    # get hf model with quantization parameters
    quantize_model auto_awq ${MODEL_PATH} ${quant_hf_model_path}

    # convert hf-awq-quant models to turbomind format
    convert_turbomind llama awq ${quant_hf_model_path} 128 ${quant_turbomind_model_path}

  elif [ $precision = kv8 ]; then
    # Convert hf model to turbomind format
    convert_turbomind llama hf ${MODEL_PATH} 0 ${quant_turbomind_model_path}

    # Get the quantization parameters and save them to the turbomind model directory
    quantize_model calibrate ${MODEL_PATH} ${quant_turbomind_model_path}/triton_models/weights
  else
    echo "Precision only supports fp16, w4a16 or kv8"
    exit
  fi
elif [ $deploy_model_format = hf ]; then
  if [ $precision = fp16 ]; then
    echo "LMDeploy supports loading HuggingFace models without model conversion."
    exit

  elif [ $precision = w4a16 ]; then
    transformers_version=$(grep 'transformers_version' ${MODEL_PATH}/config.json | sed 's/.*"transformers_version": "\(.*\)",/\1/')
    transformers_base_version=$(python3 -c "from packaging.version import parse; print(parse('${transformers_version}').base_version)")
    python3 -m pip install --no-cache-dir transformers==${transformers_base_version}

    quantize_model auto_awq ${MODEL_PATH} ${quant_hf_model_path}

  elif [ $precision = kv8 ]; then
    cp -r ${MODEL_PATH} ./${quant_hf_model_path}

    # Get the quantization parameters and save them to the hf model directory
    quantize_model calibrate ${quant_hf_model_path} ${quant_hf_model_path}
  else
    echo "Precision only supports fp16, w4a16 or kv8"
    exit
  fi
else
  echo "Deploying model only supports turbomind or hf format"
  exit
fi
