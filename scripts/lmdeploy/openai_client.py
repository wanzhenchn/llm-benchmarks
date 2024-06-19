################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-02 16:06:59
# @Details  : Examples for OpenAI's Completions and Chat API with LMDeploy's HTTP server
################################################################################

from typing import Any, Dict, Iterable, List, Optional, Union
import fire
from openai import OpenAI


class OpenAIClient:
    def __init__(self, server_addr: str):
        # Modify OpenAI's API key and API base to use vLLM's API server.
        self.openai_api_key = "EMPTY"
        self.openai_api_base = f'{server_addr}/v1'

        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )
        self.model_name = self.get_model_name()

    def get_model_name(self):
        models = self.client.models.list()
        model_name = models.data[0].id
        return model_name

    def chat_completion(self,
                        messages: Union[str, List[Dict[str, str]]],
                        max_tokens: int = 256,
                        repetition_penalty: float = 1.15,
                        temperature: float = 0.7,
                        top_p: float = 0.95,
                        top_k: int = 3,
                        stream: bool = False
                        ):
        res = self.client.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            model=self.model_name,
            extra_body={
                'top_k': top_k,
                'repetition_penalty': repetition_penalty,
            },
        )
        if stream:
            for chunk in res:
                out = chunk.choices[0].delta.content
                usage = getattr(chunk, 'usage', '')
                finish_reason = chunk.choices[0].finish_reason
                yield out, usage, finish_reason
        else:
            out = res.choices[0].message.content
            usage = dict(res.usage)
            finish_reason = res.choices[0].finish_reason
            yield out, usage, finish_reason

    def completion(self,
                   prompt: str,
                   max_tokens: int = 256,
                   repetition_penalty: float = 1.15,
                   temperature: float = 0.7,
                   top_p: float = 0.95,
                   top_k: int = 3,
                   stream: bool = False
                   ):
        res = self.client.completions.create(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            model=self.model_name,
            extra_body={
                'top_k': top_k,
                'repetition_penalty': repetition_penalty,
            },
        )
        if stream:
            for chunk in res:
                out = chunk.choices[0].text
                usage = dict(chunk.usage) if chunk.usage else chunk.usage
                finish_reason = chunk.choices[0].finish_reason
                yield out, usage, finish_reason
        else:
            out = res.choices[0].text
            usage = dict(res.usage)
            finish_reason = res.choices[0].finish_reason
            yield out, usage, finish_reason


def main(service_name: str = '0.0.0.0',
         service_port: str = '80',
         chat_enable: bool = False,
         stream: bool = False,
         ):
    server_addr = f'http://{service_name}:{service_port}'
    client = OpenAIClient(server_addr)

    request_output_len = 256

    top_k = 3
    top_p = 0.95
    temperature = 0.7
    repetition_penalty = 1.15

    prompt = "Please introduce Beijing in detail."
    messages = [
#        {"role": "system",
#         "content": "You are a helpful assistant."
#         },
        {"role": "user",
         "content": prompt,
        }
    ]

    if chat_enable:
        completion_func = client.chat_completion
        """ for llama model, the /v1/chat/completions API only accepts a single
        prompt, please refer to: https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/model.py#L130-L143
        """
        inputs = prompt
#        inputs = messages
    else:
        completion_func = client.completion
        inputs = prompt

    print(f"{'Chat completion' if chat_enable else 'completion'} results:")
    for out, usage, finish_reason in completion_func(
        inputs, request_output_len, repetition_penalty,
        temperature, top_p, top_k, stream
    ):
        if stream:
            print(out or '', end='', flush=True)
        else:
            print(out)
    if stream:
        print(f'\nfinish reson: {finish_reason}')
    else:
        print(f'prompt tokens: {usage["prompt_tokens"]}, ' +
              f'generated tokens: {usage["completion_tokens"]},' +
              f'finish reason: {finish_reason}')


if __name__ == "__main__":
    fire.Fire(main)
