<div align="center">

LLM-Benchmarks
===========================
<h4> A Benchmark Toolbox for LLM Performance (Inference and Evalution).</h4>

[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

---
<div align="left">

## Latest News 🔥
- [2024/07/04] Support for evaluation with [vLLM](https://github.com/vllm-project/vllm/) backend using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- [2024/06/21] Added support for inference performance benchmark with [LMDeploy](https://github.com/InternLM/lmdeploy) and [vLLM](https://github.com/vllm-project/vllm/).
- [2024/06/14] Added support for inference performance benchmark with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- [2024/06/14] We officially released LLM-Benchmarks!


## LLM-Benchmarks Overview

LLM-Benchmarks is an easy-to-use toolbox for benchmarking Large Language Models (LLMs) performance on inference and evalution.

- Inference Performance: Benchmarking LLMs service deployed with inference frameworks (e.g., [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [lmdeploy](https://github.com/InternLM/lmdeploy) and [vLLM](https://github.com/vllm-project/vllm),) under different batch sizes and generation lengths.

- Task Evaluation: Few-shot evaluation of LLMs throuth APIs including [OpenAI](https://openai.com/), and [Triton Inference Server](https://github.com/triton-inference-server) with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).


## Getting Started

### Download the ShareGPT dataset

You can download the dataset by running:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Prepare for Docker image and container environment

You can build Docker images locally by running:
```bash
# for tensorrt-llm
bash scripts/trt_llm/build_docker.sh all

# for lmdeploy
bash scripts/lmdeploy/build_docker.sh

# for vllm
bash scripts/vllm/build_docker.sh
```
or use the available images by `docker pull ${Image}:${Tag}`:

| Image                                                   | Tag                              |
|---------------------------------------------------------|----------------------------------|
| registry.cn-beijing.aliyuncs.com/devel-img/lmdeploy     | 0.5.1-arch_808990                |
| registry.cn-beijing.aliyuncs.com/devel-img/vllm         | 0.5.2-arch_70808990              |
| registry.cn-beijing.aliyuncs.com/devel-img/tensorrt-llm | 0.12.0.dev2024071600-arch_808990 |


### Run benchmarks

- Inference Performance
```bash
# Please confirm the version of the image used in the script
bash run_benchmark.sh model_path dataset_path sample_num device_id(like 0 or 0,1)

```

- Task Evaluation
```bash
# Build evalution image
bash scripts/evaluation/build_docker.sh vllm # (or lmdeploy or trt-llm)

# Evalution with vLLM backend
bash run_eval.sh mode(fp16, fp8-kv-fp16, fp8-kv-fp8) model_path device_id(like 0 or 0,1)"
```
