################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-06-12 14:23:45
# @Details  : client for tensorrt-llm deployed with triton service
################################################################################

import json
import requests
import fire

# Credit to: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html#
#            https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md
def make_request(api_url: str,
                 prompt: str,
                 request_output_len: int,
                 stream: bool,
                 top_k: int,
                 top_p: float,
                 temperature: float,
                 repetition_penalty: float):
    pload = {
        'text_input': prompt,
        'max_tokens': request_output_len,
        'stream': stream,
        'accumulate_tokens': True,
        'top_p': top_p,
        'top_k': top_k,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
    }

    response = requests.post(url=api_url, json=pload, stream=stream)

    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b'\n'):
        if chunk:
            decoded = chunk.decode('utf-8')
            if stream:
                if decoded.startswith('data: '):
                    decoded = decoded[len('data: '):]
            data = json.loads(decoded)
            output = data["text_output"]
            yield output


def main(service_name: str = '0.0.0.0',
         http_port: str = '800',
         max_tokens: int = 256,
         stream: bool = False):
    base_url = f"http://{service_name}:{http_port}/v2/models/ensemble/"
    if stream:
        api_url = base_url + "generate_stream"
    else:
        api_url = base_url + "generate"

    prompt = "What is machine learning?"
    top_k = 3
    top_p = 0.95
    temperature = 1e-7
    repetition_penalty = 1.15

    for output in make_request(api_url,
                               prompt,
                               max_tokens,
                               stream,
                               top_k,
                               top_p,
                               temperature,
                               repetition_penalty
                               ):
        if stream:
            print(output or '', end='', flush=True)
        else:
            print(output)


if __name__ == "__main__":
    fire.Fire(main)
