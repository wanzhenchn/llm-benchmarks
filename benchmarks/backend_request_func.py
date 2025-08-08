# Credit to: https://github.com/vllm-project/vllm/blob/main/benchmarks/backend_request_func.py

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Any, Union

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: Union[str, List[Any]]
    api_url: str
    model: str
    request_output_len: int
    top_p: float
    top_k: int
    repetition_penalty: float
    temperature: float
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[List[dict]] = None


@dataclass
class RequestFuncOutput:
    prompt: str = ""
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    output_len: int = 0
    error: str = ""


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "max_tokens": request_func_input.request_output_len,
            "stream": True,
            "temperature": request_func_input.temperature,
            "top_p": request_func_input.top_p,
            "top_k": request_func_input.top_k,
            "repetition_penalty": request_func_input.repetition_penalty,
        }
        extra_body = request_func_input.extra_body
        if extra_body:
            if extra_body.get("ignore_eos", False):
                payload["min_length"] = request_func_input.request_output_len
                extra_body.pop("ignore_eos")
            payload.update(extra_body)
        output = RequestFuncOutput()
        output.prompt = request_func_input.prompt
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = remove_prefix(chunk.decode("utf-8"), "data: ")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "v1/completions"
    ), "OpenAI Completions API URL must end with 'v1/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.request_output_len,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
            "temperature": request_func_input.temperature,
            "top_p": request_func_input.top_p,
            "top_k": request_func_input.top_k,
            "repetition_penalty": request_func_input.repetition_penalty,
        }
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt = request_func_input.prompt

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = remove_prefix(chunk.decode("utf-8"), "data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here, e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            if usage := data.get("usage"):
                                # get usage summary response
                                output.prompt_len = usage.get("prompt_tokens")
                                output.output_len = usage.get("completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = ("Never received a valid chunk to calculate TTFT."
                                        "This response will be marked as failed")

                    # get usage summary response
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.extend(request_func_input.multi_modal_content)
        payload = {
            "model": request_func_input.model,
            "messages": [
                {"role": "user", "content": content},
            ],
            "max_tokens": request_func_input.request_output_len,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
            "temperature": request_func_input.temperature,
            "top_p": request_func_input.top_p,
            "top_k": request_func_input.top_k,
            "repetition_penalty": request_func_input.repetition_penalty,
        }
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt = request_func_input.prompt

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = remove_prefix(chunk.decode("utf-8"), "data: ")
                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += content or ""
                            if usage := data.get("usage"):
                                # get usage summary response
                                output.prompt_len = usage.get("prompt_tokens")
                                output.output_len = usage.get("completion_tokens")

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


# we can't use str.removeprefix(prefix) introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_chat_completions,
    "lmdeploy": async_request_openai_chat_completions,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "trt-llm": async_request_openai_chat_completions,
}
