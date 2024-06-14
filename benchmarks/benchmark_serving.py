################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-15 10:47:20
# @Details  : benchmark script
################################################################################

import json
import os
import random
import time
import asyncio
import requests
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
import tabulate
import fire
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tokenizer import get_tokenizer


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    input_token_avg: float
    total_output: int
    output_token_avg: float
    elapsed_time: float
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft: float
    median_ttft: float
    p99_ttft: float
    mean_tpot: float
    median_tpot: float
    p99_tpot: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    avg_latency: float


def read_dataset(dataset_name: str, dataset_path: str, num_requests: int,
                 tokenizer: PreTrainedTokenizerBase) -> List[Tuple[str, int]]:
    start = time.perf_counter()
    # for ShareGPT dataset
    if dataset_name is not None:
        assert dataset_name == "sharegpt", "Only support the ShareGPT dataset"
        with open(dataset_path, 'r') as f:
            prompts = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in prompts
                   if len(data["conversations"]) >= 2]
        prompts = [data["conversations"][0]["value"] for data in dataset]
    else:
        if dataset_path.endswith("csv"):
            df = pd.read_csv(dataset_path, index_col=None)
            prompts = list(df['prompt'])
        elif dataset_path.endswith("json"):
            with open(dataset_path, 'r') as fp:
                prompts = json.load(fp)
        else:
            assert False, f"{dataset_path} format not supported"

    logging.info(f'elapsed time for read {dataset_path}: '
          f'{round(time.perf_counter() - start, 2)} s')

    num_req = min(num_requests, len(prompts))
    if num_req > 0:
        dataset_sampled = random.sample(prompts, num_req)
    else:
        logging.error('samples number should > 0')

    filtered_dataset: List[Tuple[str, int]] = []
    # Tokenize the prompts
    for prompt in dataset_sampled:
        prompt_token_ids = tokenizer(prompt).input_ids
        prompt_len = len(prompt_token_ids)
        filtered_dataset.append((prompt, prompt_len))
    return filtered_dataset


def calculate_metrics(outputs: List[RequestFuncOutput],
                      dur_s: float,
                      tokenizer: PreTrainedTokenizerBase,
                      ) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens = []
    total_input = 0
    completed = 0
    tpots = []
    ttfts = []
    res_latency = []

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = len(tokenizer(outputs[i].generated_text).input_ids)
            actual_output_lens.append(output_len)
            res_latency.append(outputs[i].latency)
            total_input += outputs[i].prompt_len
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            ttfts.append(outputs[i].ttft)
            completed += 1
        else:
            actual_output_lens.append(0)
            res_latency.append(0)

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        input_token_avg=total_input / completed,
        total_output=sum(actual_output_lens),
        output_token_avg=sum(actual_output_lens) / completed,
        elapsed_time=dur_s,
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        # ttfts is empty if streaming is not supported by backend
        mean_ttft=np.mean(ttfts or 0),
        median_ttft=np.median(ttfts or 0),
        p99_ttft=np.percentile(ttfts or 0, 99),
        mean_tpot=np.mean(tpots),
        median_tpot=np.median(tpots),
        p99_tpot=np.percentile(tpots, 99),
        p90_latency=np.percentile(res_latency, 90),
        p95_latency=np.percentile(res_latency, 95),
        p99_latency=np.percentile(res_latency, 99),
        avg_latency=np.mean(res_latency),
    )

    return metrics, actual_output_lens


async def warmup(model_id: str,
                 api_url: str,
                 request_func: Callable,
                 max_concurrency: int,
                 input_requests: List[Tuple[str, int]],
                 req_output_len: int = 128,
                 top_k: int = 3,
                 top_p: float = 0.95,
                 temperature: float = 0.01,
                 repetition_penalty: float = 1.15):
    logging.info('start to warmup ...')

    _start = time.perf_counter()

    input_requests = input_requests[:max_concurrency] if max_concurrency > 20 \
        else input_requests[:20]
    semaphore = asyncio.Semaphore(max_concurrency)  # Semaphore to limit concurrency
    tasks = []
    for prompt, prompt_len in input_requests:
        tasks.append(
            asyncio.create_task(
                process(semaphore, api_url, model_id, prompt, prompt_len,
                        request_func, req_output_len, top_k, top_p,
                        repetition_penalty, temperature, 1, False, None)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    _end = time.perf_counter()
    logging.info(f'end warmup, elapsed time: {round(_end - _start, 2)} s')


async def process(sem: asyncio.Semaphore,
                  api_url: str,
                  model_id: str,
                  prompt: str,
                  prompt_len: int,
                  request_func: Callable,
                  req_output_len: int,
                  top_k: int,
                  top_p: float,
                  repetition_penalty: float,
                  temperature: float,
                  best_of: int,
                  use_beam_search: bool,
                  pbar: Optional[tqdm] = None,
                  ):
    async with sem:  # Ensure only max_concurrency tasks run in parallel
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            prompt_len=prompt_len,
            api_url=api_url,
            request_output_len=req_output_len,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            best_of=best_of,
            use_beam_search=use_beam_search
            )
        # Call the request function directly here and return its result
        return await request_func(
            request_func_input=request_func_input, pbar=pbar
        )


async def benchmark(api_url: str,
                    model_id: str,
                    tokenizer: PreTrainedTokenizerBase,
                    request_func: Callable,
                    max_concurrency: int,
                    input_requests: List[Tuple[str, int]],
                    req_output_len: int,
                    top_k: int,
                    top_p: float,
                    repetition_penalty: float,
                    temperature: float,
                    best_of: int,
                    use_beam_search: bool,
                    disable_tqdm: bool):
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    semaphore = asyncio.Semaphore(max_concurrency)  # Semaphore to limit concurrency
    benchmark_start_time = time.perf_counter()

    tasks = []
    for prompt, prompt_len in input_requests:
        tasks.append(
            asyncio.create_task(
                process(semaphore, api_url, model_id, prompt, prompt_len,
                        request_func, req_output_len, top_k, top_p,
                        repetition_penalty, temperature, best_of,
                        use_beam_search, pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    logging.info(f'\n{"-" * 50}\n'
                 f'input_token_len: \n{[out.prompt_len for out in outputs]}\n\n'
                 f'output_token_len: \n{actual_output_lens}\n\n'
                 f'TTFT (s): \n{[round(out.ttft, 3) for out in outputs]}\n\n'
                 f'Latency (s): \n{[round(out.latency, 3) for out in outputs]}\n'
                 f'{"-" * 50}\n\n')

    result = {"Successful Request": metrics.completed,
              "Request_Gen_Token_Len": req_output_len,
              "Batch Size": max_concurrency,
              "Avg_Input_Token_Len": round(metrics.input_token_avg, 2),
              "Avg_Gen_Token_Len": round(metrics.output_token_avg, 2),
              "Elapse_Time (s)": round(metrics.elapsed_time, 3),
              "Time_to_First_Token_AVG (s)": round(metrics.mean_ttft, 3),
              "Time_to_First_Token_P99 (s)": round(metrics.p99_ttft, 3),
              "Time_per_Output_Token_AVG (s)": round(metrics.mean_tpot, 3),
              "Time_per_Output_Token_P99 (s)": round(metrics.p99_tpot, 3),
              "Latency_P90 (s)": round(metrics.p90_latency, 3),
              "Latency_P95 (s)": round(metrics.p95_latency, 3),
              "Latency_P99 (s)": round(metrics.p99_latency, 3),
              "Latency_AVG (s)": round(metrics.avg_latency, 3),
              "Token QPS (token/s)": round(metrics.output_throughput, 2),
              "Service QPS (req/s)": round(metrics.request_throughput, 2),
            }

    df = pd.DataFrame([result])
    df = df.transpose()
    df.columns = ["" for i in range(len(df.columns))]
    logging.info('Performance Summary' \
                 f'{df.to_markdown(tablefmt="simple", numalign="left", stralign="left")}\n')
    return result


def main(backend: str = "vllm",
         host: str = "localhost",
         port: int = 8000,
         endpoint: str = "/v1/completions",
         dataset_name: str = "sharegpt",
         dataset_path: str = None,
         batch_size: int = 1,
         num_requests: int = 200,
         request_output_len: int = 256,
         seed: int = 0,
         top_k: int = 3,
         top_p: float = 0.95,
         temperature: float = 1e-7,
         repetition_penalty: float = 1.15,
         best_of: int = 1,
         use_beam_search: bool = False,
         tokenizer_name_or_path: str = None,
         trust_remote_code: bool = True,
         disable_tqdm: bool = False,
         save_result: bool = True,
         log_path: str = 'perf.log'):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    random.seed(seed)
    np.random.seed(seed)

    api_url = f"http://{host}:{port}{endpoint}"
    api_url_available_models = f"http://{host}:{port}/v1/models"

    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if (backend == "vllm") or (backend == "lmdeploy"):
        response = requests.get(api_url_available_models)
        if response.status_code == 200:
            model_id = response.json()['data'][0]['id']
        else:
            raise ValueError(f"Error response from: {api_url_available_models}")
    else:
        assert tokenizer_name_or_path is not None
        model_id = None

    tokenizer_id = tokenizer_name_or_path if tokenizer_name_or_path is \
        not None else model_id
    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=trust_remote_code)


    input_requests = read_dataset(dataset_name, dataset_path, num_requests,
                                  tokenizer)
    input_requests = input_requests * max(1, int(batch_size / 64))

    # warmup
    _ = asyncio.run(
        warmup(model_id=model_id, api_url=api_url, request_func=request_func,
               max_concurrency=batch_size, input_requests=input_requests)
    )

    # run benchmark
    benchmark_result = asyncio.run(
        benchmark(
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            request_func=request_func,
            max_concurrency=batch_size,
            input_requests=input_requests,
            req_output_len=request_output_len,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            best_of=best_of,
            use_beam_search=use_beam_search,
            disable_tqdm=disable_tqdm,
        ))

    # Save results to csv file
    if save_result:
        csv_file_path = os.path.splitext(log_path)[0] + ".csv"
        header = "Successful Request, Request_Gen_Token_Len, " \
                 "Batch Size, Avg_Input_Token_Len, " \
                 "Avg_Gen_Token_Len, Elapse_Time (s), " \
                 "Time_to_First_Token_AVG (s), Time_to_First_Token_P99 (s), " \
                 "Time_per_Output_Token_AVG (s), Time_per_Output_Token_P99 (s), " \
                 "Latency_P90 (s), Latency_P95 (s), " \
                 "Latency_P99 (s), Latency_AVG (s), " \
                 "Token QPS (token/s), Service QPS (req/s)\n"
        with open(csv_file_path, 'a') as f:
            if not f.tell():
                f.write(header)
            line = ''.join(str(v)+',' for v in benchmark_result.values()) + '\n'
            f.write(line)
            logging.info(f'Performance data have been saved in {csv_file_path}')


if __name__ == "__main__":
    fire.Fire(main)
