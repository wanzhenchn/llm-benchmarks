################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-03-20 18:02:51
# @Details  : client for lmdpeloy deployed with openai service
################################################################################

from typing import Any, Dict, Iterable, List, Optional, Union
import json
import base64
import requests
import fire
import logging

# Credit to: https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/api_client.py
def json_loads(content):
    """Loads content to json format."""
    try:
        content = json.loads(content)
        return content
    except:  # noqa
        logging.warning(f'weird json content {content}')
        return ''


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class APIClient:
    def __init__(self, server_addr: str):
        self.api_server_url = f'{server_addr}/v1/chat/completions'
        self._models_v1_url = f'{server_addr}/v1/models'
        self.model_name = self.get_model_list(self._models_v1_url)[0]

    @staticmethod
    def get_model_list(api_url: str):
        """Get model list from api server."""
        response = requests.get(api_url)
        if hasattr(response, 'text'):
            model_list = json.loads(response.text)
            model_list = model_list.pop('data', [])
            return [item['id'] for item in model_list]
        return None

    def chat_completions_v1(self,
                            messages: Union[str, List[Dict[str, str]]],
                            request_output_len: Optional[int] = None,
                            top_k: Optional[int] = 3,
                            top_p: Optional[float] = 0.95,
                            temperature: Optional[float] = 0.7,
                            repetition_penalty: Optional[float] = 1.15,
                            stream: Optional[bool] = False,
                            **kwargs):
        headers = {'content-type': 'application/json'}
        pload = {
        #    messages: string prompt or chat history in OpenAI format.
        #    Chat history example: `[{"role": "user", "content": "hi"}]`.
            'model': self.model_name,
            'messages': messages,
            'stream': stream,
            'max_tokens': request_output_len,
            'top_k': top_k,
            'top_p': top_p,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty,
            'stop': None,
        }

        response = requests.post(url=self.api_server_url,
                                 headers=headers,
                                 json=pload,
                                 stream=stream)

        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b'\n'):
            if chunk:
                decoded = chunk.decode('utf-8')
                if stream:
                    if decoded == 'data: [DONE]':
                        continue
                    if decoded[:6] == 'data: ':
                        decoded = decoded[6:]
                data = json_loads(decoded)
                if stream:
                    output = data['choices'][0]['delta'].pop('content', '')
                else:
                    output = data['choices'][0]['message'].pop('content', '')
                usage = data.get('usage','')
                finish_reason = data['choices'][0]['finish_reason']
                yield output, usage, finish_reason


def main(model_type: str = 'llm',
         service_name: str = '0.0.0.0',
         service_port: str = '80',
         stream: bool = False,
         ):
    server_addr = f'http://{service_name}:{service_port}'
    api_client = APIClient(server_addr)

    request_output_len = 256
    top_k = 3
    top_p = 0.95
    temperature = 0.7
    repetition_penalty = 1.15

    if model_type == 'llm':
        messages = "I want to go to Beijing for vacation, please give me a plan."
    else:
        prompt_list = [
            'Describe the image please',
            'What is unusual about this image?',
        ]

        img_url_list = [
            'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
#            'https://www.barnorama.com/wp-content/uploads/2016/12/03-Confusing-Pictures.jpg',
            './data/03-Confusing-Pictures.jpg'
        ]

        messages = [
            {'role': 'user',
             'content': [{'type': 'text',
                          'text': prompt_list[1],
                          },
                         {'type': 'image_url',
                          'image_url': {'url': f"data:image/jpeg;base64," +
                                        f"{encode_image(img_url_list[1])}",
                                        }
#                          'image_url': {'url': img_url_list[1],},
                          }
                         ]
             }
        ]

    for output, usage, finish_reason in api_client.chat_completions_v1(
        messages=messages,
        request_output_len=request_output_len,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        stream=stream):
        if stream:
            print(output or '', end='', flush=True)
        else:
            print(output)
            print(f'prompt tokens: {usage["prompt_tokens"]}, ' +
                  f'generated tokens: {usage["completion_tokens"]}, ' +
                  f'finish reson: {finish_reason}')
    if stream:
        print(f'\nfinish reson: {finish_reason}')


if __name__ == "__main__":
    fire.Fire(main)
