#!/bin/bash
################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-22 18:47:18
# @Details  : Create triton model template and modify parameters in conifg.pbtxt
################################################################################

set -euxo pipefail

# https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md
if [ $# != 3  ]; then
  echo "Usage: $0 triton_model_repo_path tokenizer_path max_batch_size"
  exit
fi
triton_model_dir=$1
tokenizer_dir=$2
max_batch_size=$3


if [ ! -d ${triton_model_dir} ]; then
  cp -r /app/triton_model_repo_template ${triton_model_dir}
fi

fill_template=/app/tools/fill_template.py

triton_max_batch_size=${max_batch_size}
kv_cache_free_gpu_mem_fraction=0.9
max_beam_width=1
max_queue_delay_microseconds=0

engine_path=${triton_model_dir%/}/tensorrt_llm/1
engine_config_path=${triton_model_dir%/}/tensorrt_llm/config.pbtxt
preprocess_config_path=${triton_model_dir%/}/preprocessing/config.pbtxt
postprocess_config_path=${triton_model_dir%/}/postprocessing/config.pbtxt
ensemble_config_path=${triton_model_dir%/}/ensemble/config.pbtxt
bls_config_path=${triton_model_dir%/}/tensorrt_llm_bls/config.pbtxt

# copy tokenizer model to the target path
cp ${tokenizer_dir%/}/tokenizer* ${triton_model_dir}

# fill config.pbtxt
python ${fill_template} --in_place ${engine_config_path} \
  triton_max_batch_size:${triton_max_batch_size},triton_backend:tensorrtllm,batching_strategy:inflight_fused_batching,engine_dir:${engine_path},batch_scheduler_policy:max_utilization,decoupled_mode:True,kv_cache_free_gpu_mem_fraction:${kv_cache_free_gpu_mem_fraction},max_beam_width:${max_beam_width},max_queue_delay_microseconds:${max_queue_delay_microseconds}

python ${fill_template} --in_place ${preprocess_config_path} \
  tokenizer_dir:${triton_model_dir},triton_max_batch_size:${triton_max_batch_size},preprocessing_instance_count:1

python ${fill_template} --in_place ${postprocess_config_path} \
  tokenizer_dir:${triton_model_dir},triton_max_batch_size:${triton_max_batch_size},postprocessing_instance_count:1

python ${fill_template} --in_place ${ensemble_config_path} \
  triton_max_batch_size:${triton_max_batch_size}

python ${fill_template} --in_place ${bls_config_path} \
  triton_max_batch_size:${triton_max_batch_size},decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
