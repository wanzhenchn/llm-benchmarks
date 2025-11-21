################################################################################
# @Copyright: All Rights Reserved.
# @Author   : wanzhenchn@gmail.com
# @Date     : 2025-10-14 16:06:59
# @Details  : Demonstrates the usage of cpu offloading with AIS KVCache/LMCache in vLLM v1.
################################################################################

import fire
import contextlib
import os
import time
from typing import Literal
from dataclasses import asdict

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs


@contextlib.contextmanager
def build_llm_with_kvcache(cache_connector: str,
                           model: str,
                           tp: int = 1,
                           enable_prefix_caching: bool = False):
    ktc = KVTransferConfig(
        kv_connector=cache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp,
        enable_prefix_caching=enable_prefix_caching,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        if "LMCache" in cache_connector:
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.cache_engine import LMCacheEngineBuilder

            # Clean up lmcache backend
            LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[str],
    sampling_params: SamplingParams,
    req_str: str,
):
    # Should be able to see logs like the following:
    # `LMCache/AIS KVCache INFO: Storing KV cache for 6006 out of 6006 tokens for request 0`
    # This indicates that the KV cache has been stored in LMCache/AIS KVCache.
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)



def main(connector: str = "lmcache",
         tp: int = 1,
         enable_prefix_caching: bool = False,
         model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
         ):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if "lmcache" in connector.lower():
        cache_connector = "LMCacheConnectorV1"

        # LMCache-related environment variables
        # Use experimental features in LMCache
        os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
        # LMCache is set to use 256 tokens per chunk
        os.environ["LMCACHE_CHUNK_SIZE"] = "256"
        # Enable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        # Set local CPU memory limit to 5.0 GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

    else:
        raise ValueError('Only support lmcache or ais_kvcache connector')


    with build_llm_with_kvcache(
        cache_connector, model, tp, enable_prefix_caching
    ) as llm:
        # This example script runs two requests with a shared prefix.
        # Define the shared prompt and specific prompts
        shared_prompt = "Hello, how are you?" * 1000
        first_prompt = [
            shared_prompt + "Hello, my name is",
        ]
        second_prompt = [
            shared_prompt + "Tell me a very long story",
        ]

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

        # Print the first output
        print_output(llm, first_prompt, sampling_params, "first")

        time.sleep(1)

        # print the second output
        print_output(llm, second_prompt, sampling_params, "second")


if __name__ == "__main__":
    fire.Fire(main)
