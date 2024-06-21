<div align="center">

LLM-Benchmarks
===========================
<h4> A Benchmark Toolbox for LLM Performance (Inference and Evalution).</h4>

[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

---
<div align="left">

## Latest News 🔥
- [2024/06/21] Added support for inference performance benchmark with [LMDeploy](https://github.com/InternLM/lmdeploy).
- [2024/06/14] Added support for inference performance benchmark with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- [2024/06/14] We officially released LLM-Benchmarks!


## LLM-Benchmarks Overview

LLM-Benchmarks is an easy-to-use toolbox for benchmarking Large Language Models (LLMs) performance on inference and evalution.

- Inference Performance: Benchmarking LLMs service deployed with inference frameworks (e.g., [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [lmdeploy](https://github.com/InternLM/lmdeploy) and [vLLM](https://github.com/vllm-project/vllmA),) under different batch sizes and generation lengths.

- Task Evaluation: Few-shot evaluation of LLMs throuth APIs including [OpenAI](https://openai.com/), and [Triton Inference Server](https://github.com/triton-inference-server) with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).


## Getting Started

### Download the ShareGPT dataset

You can download the dataset by running:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Prepare for Docker image and container environment

You can build docker images by running:
```bash
# for tensorrt-llm
bash scripts/trt_llm/build_docker.sh all

# for lmdeploy
bash scripts/lmdeploy/build_docker.sh

# for vllm
bash scripts/vllm/build_docker.sh
```

### Run benchmarks

- Inference Performance
```bash
bash run_benchmark.sh model_path dataset_path sample_num device_id(0 or 0,1)

```

- Task Evaluation
