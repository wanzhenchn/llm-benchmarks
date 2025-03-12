################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2025-03-10 14:36:54
# @Details  : A workaround to pad intermediate_size from 29568 to 29568 to ena
#             enable multi-TP execution for the Qwen2.5-72B(-Instruct) AWQ-quantized model.
################################################################################
import fire
import os
from tqdm import tqdm
from typing import Dict

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# credit to: https://qwen.readthedocs.io/en/latest/quantization/gptq.html#troubleshooting
#            https://github.com/QwenLM/Qwen2.5/issues/578

def padding_and_saving_weight(model_path: str, output_dir: str):
    # must use AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_fast=False,
                                              trust_remote_code=True)
    # this size is Qwen2.5-72B only
    pad_size = 128

    sd = model.state_dict()
    for k, v in tqdm(sd.items(), desc="Padding Weights"):
        # interleaving the padded zeros
        if ('mlp.up_proj.weight' in k) or ('mlp.gate_proj.weight' in k):
            prev_v = F.pad(v.unsqueeze(1), (0, 0, 0, 1, 0, 0)).reshape(29568*2, -1)[:pad_size*2]
            new_v = torch.cat([prev_v, v[pad_size:]], dim=0)
            sd[k] = new_v
        elif 'mlp.down_proj.weight' in k:
            prev_v= F.pad(v.unsqueeze(2), (0, 1)).reshape(8192, 29568*2)[:, :pad_size*2]
            new_v = torch.cat([prev_v, v[:, pad_size:]], dim=1)
            sd[k] = new_v

    config = model.config
    # modify the intermediate_size to 29696
    if config.intermediate_size == 29568:
        config.intermediate_size += pad_size
#    config.to_json_file(f'{output_dir}/config.json')

    # save tokenizer model
    tokenizer.save_pretrained(output_dir)

    #save weights
    model.save_pretrained(output_dir, state_dict=sd, max_shard_size="4GB")
    print(f"The padded model has been saved in {output_dir}.")


def main(model_path: str, output_dir: str = None):
    if not output_dir:
        output_dir = os.path.basename(model_path) + '-padded'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    padding_and_saving_weight(model_path, output_dir)


if __name__ == "__main__":
    fire.Fire(main)
