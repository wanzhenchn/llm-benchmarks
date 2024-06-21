"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      output_token_len: int = 256,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L25
    pload = {
        "prompt": prompt,
        "top_p": 0.95,
        "top_k": 3,
        "repetition_penalty": 1.15,
        "temperature": 0.7,
        "max_tokens": output_token_len,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=stream)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="Please introduce Beijing in detailed")
    parser.add_argument("--output-token-len", type=int, default=20)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}/generate"

    print(f"Prompt: {args.prompt!r}\n", flush=True)
    response = post_http_request(args.prompt, api_url, args.output_token_len, args.stream)

    if args.stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"{line!r}", flush=True)
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            print(f"{line!r}", flush=True)
