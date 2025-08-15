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
import base64
from PIL import Image
from typing import List, Tuple, Optional, Callable, Any, Union, AsyncGenerator
import numpy as np
import pandas as pd
import tabulate
import fire
from tqdm.asyncio import tqdm

from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)


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
    decode_throughtput_per_user: float
    mean_ttft: float
    median_ttft: float
    p99_ttft: float
    mean_tpot: float
    median_tpot: float
    p99_tpot: float
    mean_itl: float
    median_itl: float
    p99_itl: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    avg_latency: float
    gpu_util: float
    gpu_mem: float
    tensor_active: float
    sm_active: float
    sm_occupancy: float
    dram_active: float


def download_image(url:str, cache_dir:str="image_cache", retry:int=3):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    file_name = url.split('/')[-1]
    file_path = os.path.join(cache_dir, file_name)
    if os.path.exists(file_path):
        logging.info(f"Image already exists in cache: {file_path}")
        with open(file_path, "rb") as fd:
            image_bin = fd.read()
    else:
        mtries = retry
        while mtries > 0:
            try:
                response = requests.get(
                    url, stream=True, headers={"Connection": "close"}, timeout=10)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    image_bin = response.content
                    break
            except Exception as e:
                mtries -= 1
        else:
            raise Exception(f"Failed to retrieve image from URL {url}")

    image = Image.open(BytesIO(image_bin))
    buffered = BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    width, height = image.size
    return img_base64, width, height


def save_to_jsonl(jsonl_path: str, outputs: List[RequestFuncOutput]):
    data = []
    for i in range(len(outputs)):
        if outputs[i].success:
            prompt = outputs[i].prompt
            prompt_len = outputs[i].prompt_len
            output_len = outputs[i].output_len
            generated_text = outputs[i].generated_text
            data.append(
                {"prompt": prompt,
                 "generated_text": generated_text,
                 "prompt_len": prompt_len,
                 "output_len": output_len,
                }
            )
    with open(jsonl_path, 'a') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    logging.info(f'Requested len(data) results have been saved in {jsonl_path}')


def expand_dataset(input_requests: List[Tuple[str, Union[List[dict], None]]],
                   batch_size: int, enable_expand_dataset: bool = False):
    ratio = 1
    expand_input_requests = input_requests
    dataset_num = len(input_requests)
    if enable_expand_dataset:
        logging.info('Set enable_expand_dataset = True')
        while dataset_num < 3 * batch_size:
            ratio *= 2
            expand_input_requests = input_requests * ratio
            dataset_num = len(expand_input_requests)
        if ratio > 1:
            logging.info(f'Expand the dataset by {ratio} times for batch_size={batch_size}.')
            logging.info(f'(After expanded) Requests #dataset: {len(expand_input_requests)}')
            logging.warning('If the service has enabled the prefix-caching feature, '\
                            'expanding the dataset may have a certain impact on '\
                            'test data of efficiency evaluation.')
        else:
            logging.warning(f'The dataset num {len(expand_input_requests)} is enough '
                            f'for batch size {batch_size}, it does not need to be expanded.')
    return expand_input_requests


def read_dataset(dataset_name: str,
                 dataset_path: str) -> List[Tuple[str, None]]:
    start = time.perf_counter()
    # for ShareGPT dataset
    if dataset_name is not None:
        assert dataset_name == "sharegpt", "Only support the ShareGPT dataset"
        with open(dataset_path, 'r', encoding='utf-8') as f:
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
            with open(dataset_path, 'r', encoding='utf-8') as fp:
                prompts = json.load(fp)
        elif dataset_path.endswith("xlsx"):
            df = pd.read_excel(dataset_path, index_col=None)
            prompts = list(df['question'])
        elif dataset_path.endswith("jsonl") or dataset_path.endswith("txt"):
            with open(dataset_path, 'r', encoding='utf-8') as fp:
                prompts = [json.loads(line)['data']['question'] \
                           for line in fp.readlines()]
        else:
            assert False, f"{dataset_path} format not supported"

    logging.info(f'elapsed time for read {dataset_path}: '
                 f'{round(time.perf_counter() - start, 2)}s')

    input_requests = []
    for prompt in prompts:
        input_requests.append((prompt, None))

    logging.info(f'(Before expanded) Requests #dataset: {len(input_requests)}')
    return input_requests


def read_vlm_dataset(dataset_name: str,
                     dataset_path: str,
                     use_base64: bool=False) -> List[Tuple[str,List[dict]]]:
    start = time.perf_counter()
    if dataset_path.endswith("csv"):
        df = pd.read_csv(dataset_path, index_col=None)
        prompts = list(df['prompt'])
        image_urls = list(map(lambda x: x.split(","), list(df['image_url'])))
        samples = list(zip(prompts, image_urls))
    else:
        assert False, f"{dataset_path} format only supports csv file which " \
                      "consists of column named 'prompt' and 'image_url'."

    logging.info(f'elapsed time for read {dataset_id}: '
                 f'{round(time.perf_counter() - start, 2)}s')

    # request with img_base64 example:
    # - vllm: https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py
    # Multi-image for future, current LLM inference may not support prompt with multi-images
    input_requests = []
    for (prompt, image_urls) in samples:
        mm_content = []
        for image_url in image_urls:
            img_base64, _, _ = download_image(image_url)
            mm_content.append(
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f"{image_url}" if not use_base64 else f'data:image/jpeg;base64,{img_base64}'
                    }
                }
            )
        input_requests.append((prompt, mm_content))

    logging.info(f'(Before expanded) Requests #dataset: {len(input_requests)}')
    return input_requests


async def get_request(
    input_requests: List[Tuple[str, Union[List[dict], None]]],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[str, Union[List[dict], None]], None]:
    input_requests = iter(input_requests)
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(outputs: List[RequestFuncOutput],
                      dur_s: float,
                      gpu_metrics: dict,
                      ) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens = []
    total_input = 0
    completed = 0
    decode_throughtput: List[float] = [] # token/user/sec
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    res_latency: List[float] = []

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            actual_output_lens.append(output_len)
            total_input += outputs[i].prompt_len
            tpot = 0
            decode_tps = 0
            if output_len > 1:
                tpot = (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                decode_tps = 1.0 / tpot
            tpots.append(tpot)
            decode_throughtput.append(decode_tps)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            res_latency.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)
            res_latency.append(0)

    # check completed request
    assert completed > 0, f"The number of requests that returned successfully " \
        f"is {completed}. Please check the service logs to further diagnose the issue."

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
        decode_throughtput_per_user=np.mean(decode_throughtput),
        # ttfts is empty if streaming is not supported by backend
        mean_ttft=np.mean(ttfts or 0) * 1000,
        median_ttft=np.median(ttfts or 0) * 1000,
        p99_ttft=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot=np.mean(tpots or 0) * 1000,
        median_tpot=np.median(tpots or 0) * 1000,
        p99_tpot=np.percentile(tpots or 0, 99) * 1000,
        # add ITL results and tweak TPOT results
        mean_itl=np.mean(itls or 0) * 1000,
        median_itl=np.median(itls or 0) * 1000,
        p99_itl=np.percentile(itls or 0, 99) * 1000,
        p90_latency=np.percentile(res_latency, 90) * 1000,
        p95_latency=np.percentile(res_latency, 95) * 1000,
        p99_latency=np.percentile(res_latency, 99) * 1000,
        avg_latency=np.mean(res_latency) * 1000,
        gpu_util=np.mean(gpu_metrics['gpu_util'][3:-2] if gpu_metrics else 0),
        gpu_mem=np.mean(gpu_metrics['gpu_mem'][3:-2] if gpu_metrics else 0),
        tensor_active=np.mean(gpu_metrics['tensor_active'][3:-2] if gpu_metrics else 0),
        sm_active=np.mean(gpu_metrics['sm_active'][3:-2] if gpu_metrics else 0),
        sm_occupancy=np.mean(gpu_metrics['sm_occupancy'][3:-2] if gpu_metrics else 0),
        dram_active=np.mean(gpu_metrics['dram_active'][3:-2] if gpu_metrics else 0),
    )

    return metrics, actual_output_lens


async def process(sem: asyncio.Semaphore,
                  api_url: str,
                  model_id: str,
                  prompt: str,
                  mm_content: Union[List[dict], None],
                  request_func: Callable,
                  req_output_len: int,
                  top_k: int,
                  top_p: float,
                  repetition_penalty: float,
                  temperature: float,
                  extra_body: Optional[dict] = None,
                  pbar: Optional[tqdm] = None,
                  ):
    async with sem:  # Ensure only max_concurrency tasks run in parallel
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            multi_modal_content=mm_content,
            api_url=api_url,
            request_output_len=req_output_len,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            extra_body=extra_body,
            )
        # Call the request function directly here and return its result
        return await request_func(
            request_func_input=request_func_input, pbar=pbar
        )


async def warmup(model_id: str,
                 api_url: str,
                 request_func: Callable,
                 max_concurrency: int,
                 input_requests: List[Tuple[str, int]],
                 req_output_len: int = 128,
                 top_k: int = 3,
                 top_p: float = 0.95,
                 temperature: float = 0.01,
                 repetition_penalty: float = 1.15,
                 extra_body: Optional[dict] = None,
                 warmup_duration_s: int = 60):
    logging.info('start to warmup ...')

    _start = time.perf_counter()
    while time.perf_counter() - _start < warmup_duration_s:
        input_requests = input_requests[:max_concurrency]
        semaphore = asyncio.Semaphore(max_concurrency)  # Semaphore to limit concurrency
        tasks = []
        for prompt, mm_content in input_requests:
            tasks.append(
                asyncio.create_task(
                    process(semaphore, api_url, model_id, prompt, mm_content,
                            request_func, req_output_len, top_k, top_p,
                            repetition_penalty, temperature, extra_body, None)
                )
            )
        outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    _end = time.perf_counter()
    logging.info(f'end warmup, elapsed time: {round(_end - _start, 2)} s')


async def benchmark(api_url: str,
                    model_id: str,
                    request_func: Callable,
                    max_concurrency: int,
                    input_requests: List[Tuple[str, int]],
                    req_output_len: int,
                    top_k: int,
                    top_p: float,
                    repetition_penalty: float,
                    temperature: float,
                    extra_body: Optional[dict],
                    disable_tqdm: bool,
                    save_result: bool,
                    debug_result: bool,
                    log_path: str,
                    get_gpu_metrics: bool,
                    get_gpu_metrics_freq: int,
                    device_ids: str,
                    request_rate: float,
                    burstiness: float):
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    semaphore = asyncio.Semaphore(max_concurrency)  # Semaphore to limit concurrency
    benchmark_start_time = time.perf_counter()

    tasks = []
    async for prompt, mm_content in get_request(input_requests, request_rate, burstiness):
        tasks.append(
            asyncio.create_task(
                process(semaphore, api_url, model_id, prompt, mm_content,
                        request_func, req_output_len, top_k, top_p,
                        repetition_penalty, temperature, extra_body, pbar)
            )
        )

    # start gpu metrics collection
    if get_gpu_metrics:
        from tools import get_metrics

        stop_event = asyncio.Event()
        gpu_metrics_task = asyncio.create_task(
            get_metrics(device_ids, get_gpu_metrics_freq, stop_event))

    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    benchmark_duration = time.perf_counter() - benchmark_start_time

    gpu_metrics = None
    if get_gpu_metrics:
        stop_event.set()
        gpu_metrics = await gpu_metrics_task

    if not disable_tqdm:
        pbar.close()

    metrics, actual_output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        gpu_metrics=gpu_metrics
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
              "Time_to_First_Token_AVG (ms)": round(metrics.mean_ttft),
              "Time_to_First_Token_P99 (ms)": round(metrics.p99_ttft),
              "Time_per_Output_Token_AVG (ms)": round(metrics.mean_tpot),
              "Time_per_Output_Token_P99 (ms)": round(metrics.p99_tpot),
              "Inter_Token_Latency_AVG (ms)": round(metrics.mean_itl),
              "Inter_Token_Latency_P99 (ms)": round(metrics.p99_itl),
              "Latency_P90 (ms)": round(metrics.p90_latency),
              "Latency_P95 (ms)": round(metrics.p95_latency),
              "Latency_P99 (ms)": round(metrics.p99_latency),
              "Latency_AVG (ms)": round(metrics.avg_latency),
              "Token Throughput (token/s)": round(metrics.output_throughput, 2),
              "Service Throughput (req/s)": round(metrics.request_throughput, 2),
              "Decode Token Throughput (token/user/s)": round(
                  metrics.decode_throughtput_per_user, 2),
              "GPU UTIL": round(metrics.gpu_util, 1),
              "GPU Mem (MB)": round(metrics.gpu_mem, 2),
              "Tensor Active": round(metrics.tensor_active, 2),
              "SM Active": round(metrics.sm_active, 2),
              "SM Occupancy": round(metrics.sm_occupancy, 2),
              "DRAM Active": round(metrics.dram_active, 2),
              "Request Rate": round(request_rate, 2),
              "Burstiness": round(burstiness, 2),
            }

    # display perf data on screen
    df = pd.DataFrame([result])
    df = df.transpose()
    df.columns = ["" for i in range(len(df.columns))]
    logging.info('Performance Summary' \
                 f'{df.to_markdown(tablefmt="simple", numalign="left", stralign="left")}\n')

    if save_result:
        from tools import display_performance_table

        # save performance data to csv file
        csv_file_path = os.path.splitext(log_path)[0] + ".csv"
        header = "Successful Request,Request_Gen_Token_Len," \
                 "Batch Size,Avg_Input_Token_Len," \
                 "Avg_Gen_Token_Len,Elapse_Time (s)," \
                 "Time_to_First_Token_AVG (ms),Time_to_First_Token_P99 (ms)," \
                 "Time_per_Output_Token_AVG (ms),Time_per_Output_Token_P99 (ms)," \
                 "Latency_P90 (ms),Latency_P95 (ms)," \
                 "Latency_P99 (ms),Latency_AVG (ms)," \
                 "Token Throughput (token/s),Service Throughput (req/s)," \
                 "Decode Token Throughput (token/user/s)," \
                 "GPU UTIL,GPU Mem (MB),Tensor Active,SM Active,SM Occupancy,DRAM Active," \
                 "Request Rate,Burstiness\n"
        del result["Inter_Token_Latency_AVG (ms)"]
        del result["Inter_Token_Latency_P99 (ms)"]
        with open(csv_file_path, 'a') as f:
            if not f.tell():
                f.write(header)
            line = ''.join(str(v)+',' for v in result.values()) + '\n'
            f.write(line)

        # print summary from csv file
        display_performance_table(csv_file_path)
        logging.info(f'Performance data have been saved in {csv_file_path}')

    # save returned data to jsonl file
    if debug_result:
        jsonl_file_path = os.path.splitext(log_path)[0] + ".jsonl"
        save_to_jsonl(jsonl_file_path, outputs)


def main(host: str = "localhost",
         port: int = 8000,
         model_type: str = "llm",
         endpoint: str = "/v1/chat/completions",
         dataset_name: str = 'sharegpt',
         dataset_path: str = None,
         enable_expand_dataset: bool = False,
         batch_size: int = 1,
         request_output_len: int = 256,
         top_k: int = 3,
         top_p: float = 0.95,
         temperature: float = 0.01,
         repetition_penalty: float = 1.15,
         extra_body: Optional[dict] = None,
         trust_remote_code: bool = True,
         disable_tqdm: bool = False,
         request_rate: float = float('inf'),
         burstiness : float = 1.0,
         save_result: bool = True,
         debug_result: bool = True,
         log_path: str = 'perf.log',
         get_gpu_metrics: bool = False,
         get_gpu_metrics_freq: int = 10,
         device_ids: str = "0,1,2,3",
         ):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

    # convert args to bool
    def to_bool(value):
        return str(value).lower() == 'true'

    args = [enable_expand_dataset, trust_remote_code, disable_tqdm,
            save_result, debug_result, get_gpu_metrics]
    enable_expand_dataset, trust_remote_code, disable_tqdm, save_result, \
        debug_result, get_gpu_metrics = map(to_bool, args)

    if "0.0.0.0" in host:
        api_url = f"http://{host}:{port}{endpoint}"
        api_url_available_models = f"http://{host}:{port}/v1/models"
    else:
        if host.endswith('/'):
            host = host[:-1]
        api_url = f"http://{host}{endpoint}"
        api_url_available_models = f"http://{host}/v1/models"

    response = requests.get(api_url_available_models)
    if response.status_code == 200:
        model_id = response.json()['data'][0]['id']
    else:
        raise ValueError(f"Error ({response.reason}) response from {api_url_available_models} "
                         f"to get model id informations.")

    if model_type == "vlm":
        input_requests = read_vlm_dataset(dataset_name, dataset_path,
                                          use_base64=True)
    elif model_type == "llm":
        input_requests = read_dataset(dataset_name, dataset_path)
    else:
        assert model_type in ["vlm", "llm"], f"{model_type} not in [llm, vlm]"

    # Expand the dataset if its num less than 3 times the request batch size
    expand_input_requests = expand_dataset(
        input_requests, batch_size, enable_expand_dataset
    )

    if endpoint.endswith("chat/completions"):
        request_func = ASYNC_REQUEST_FUNCS.get("openai-chat")
    else:
        request_func = ASYNC_REQUEST_FUNCS.get("openai")

    # warmup
    _ = asyncio.run(
        warmup(model_id=model_id, api_url=api_url, request_func=request_func,
               max_concurrency=batch_size, input_requests=input_requests)
    )

    # run benchmark
    asyncio.run(
        benchmark(api_url=api_url,
                  model_id=model_id,
                  request_func=request_func,
                  max_concurrency=batch_size,
                  input_requests=expand_input_requests,
                  req_output_len=request_output_len,
                  top_k=top_k,
                  top_p=top_p,
                  repetition_penalty=repetition_penalty,
                  temperature=temperature,
                  extra_body=extra_body,
                  disable_tqdm=disable_tqdm,
                  save_result=save_result,
                  debug_result=debug_result,
                  log_path=log_path,
                  get_gpu_metrics=get_gpu_metrics,
                  get_gpu_metrics_freq=get_gpu_metrics_freq,
                  device_ids=device_ids,
                  request_rate=request_rate,
                  burstiness=burstiness)
    )


if __name__ == "__main__":
    fire.Fire(main)
