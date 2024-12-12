#!/usr/bin/env bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-09-25 16:19:07
# @Details  : run llama_triton_example
################################################################################

set -euxo pipefail

if [ $# != 5 ]; then
  echo "Usage: $0 turbomind_model_path llama_config_path device_id prompt request_output_len"
  exit
fi

model_path=${1%/}
llama_config_path=$2
dev_id=$3
prompt=$4
request_output_len=$5

tokenizer_model_path=${model_path}/triton_models/tokenizer/tokenizer.model
weight_path=${model_path}/triton_models/weights

export CUDA_VISIBLE_DEVICES=${dev_id}

tokenizer_file="tokenizer.py"
tokenizer_file_url="https://raw.githubusercontent.com/InternLM/lmdeploy/refs/heads/main/examples/cpp/llama/tokenizer.py"
start_ids_csv="start_ids.csv"


function replace_value() {
  local config_path=$1
  local key=$2
  local value=$3
  sed -i "s|^\s*$key:.*|    $key: $value|" $config_path
}


# 1. get tokenizer.py
if [ ! -f "$tokenizer_file" ]; then
  echo "$tokenizer_file doesn't exist, downloading now..."
  wget "$tokenizer_file_url" -O "$tokenizer_file"
fi

# 2. get start_ids.csv
python3 ${tokenizer_file}  \
  --model_file ${tokenizer_model_path} \
  --encode_line "${prompt}"

# 3. modify llama_config.yaml to update parameters
key=("model_dir" "request_output_len")
value=("${weight_path}" "${request_output_len}")
if [ -f "$llama_config_path" ]; then
  for ((i=0; i<${#key[@]}; i++)); do
    replace_value $llama_config_path ${key[i]} ${value[i]}
  done
else
  echo "$llama_config_path doesn't exist, please pass it"
  exit 1
fi

# 4. run llama_triton_example
if [[ ! -x llama_triton_example ]]; then
  ln -s /opt/lmdeploy/install/bin/llama_triton_example .
fi

./llama_triton_example ${llama_config_path} ${start_ids_csv}

# 5. decode ids
tail -n 1 out | python3 ${tokenizer_file} --model_file ${tokenizer_model_path}
