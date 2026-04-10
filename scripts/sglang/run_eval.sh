#! /usr/bin/bash
set -euxo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <model_path> <port> <task>"
  exit 1
fi

model_path=$1
port=$2
task=$3

# mmmu_val -> lmms_eval; remaining comma-separated names -> lm_eval
if [[ ",${task}," == *",mmmu_val,""* ]]; then
  has_mmmu_val=1
else
  has_mmmu_val=0
fi
task_lm_eval=$(printf '%s' "$task" | tr ',' '\n' | grep -vxF 'mmmu_val' | paste -sd, -)

models_url="http://localhost:${port}/v1/models"
echo "Waiting for OpenAI-compatible server at ${models_url} ..." >&2
until curl -sf --connect-timeout 2 --max-time 10 "${models_url}" >/dev/null; do
  sleep 2
done
echo "Server is up; starting evaluation." >&2

if [[ -n "$task_lm_eval" ]]; then
  if python3 -c "import lm_eval" 2>/dev/null; then
    echo "lm_eval is already installed; skipping pip install." >&2
  else
    echo "lm_eval not found; installing lm_eval[api] ..." >&2
    python3 -m pip3 install "lm_eval[api]"
  fi
  lm_eval --model local-completions \
          --model_args model=${model_path},base_url=http://localhost:${port}/v1/completions,num_concurrent=64,max_retries=3,max_gen_toks=2048,tokenized_requests=False \
          --tasks "${task_lm_eval}" \
          --batch_size auto \
          --trust_remote_code
fi

if [[ "$has_mmmu_val" -eq 1 ]]; then
  if python3 -c "import lmms_eval" 2>/dev/null; then
    echo "lmms_eval is already installed; skipping pip install." >&2
  else
    echo "lmms_eval not found; installing lmms_eval ..." >&2
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    cd lmms-eval && python3 -m pip3 install . && cd .. && rm -rf lmms-eval
  fi
  python3 -m lmms_eval \
      --model=openai_compatible \
      --model_args model_version=${model_path} \
      --tasks mmmu_val \
      --batch_size 16
fi