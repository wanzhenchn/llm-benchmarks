################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2025-04-08 19:05:36
# @Details  :
################################################################################

import os
import fire
from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.models.definitions.base_qwen2_vl import BaseQwen2VLGPTQ

os.environ["PYTHONUTF8"]="1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def format_qwen2_vl_dataset(image, assistant):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "generate a caption for this image"},
            ],
        },
        {"role": "assistant", "content": assistant},
    ]


def prepare_dataset(format_func, n_sample: int = 20) -> list[list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]"
    )
    return [
        format_func(sample["url"], sample["caption"])
        for sample in dataset
    ]


def get_calib_dataset(model):
    if isinstance(model, BaseQwen2VLGPTQ):
        return prepare_dataset(format_qwen2_vl_dataset, n_sample=256)
    raise NotImplementedError(f"Unsupported MODEL: {model.__class__}")


def quantize(model_path: str,
             output_path: str,
             bit: int):
    quant_config = QuantizeConfig(bits=bit, group_size=128)

    model = GPTQModel.load(model_path, quant_config)
    calibration_dataset = get_calib_dataset(model)

    # increase `batch_size` to match gpu/vram specs to speed up quantization
    model.quantize(calibration_dataset, batch_size=8)

    model.save(output_path)

    # test post-quant inference
    model = GPTQModel.load(output_path)
    result = model.generate("Uncovering deep insights begins with")[0] # tokens
    print(model.tokenizer.decode(result)) # string output


if __name__ == "__main__":
    fire.Fire(quantize)
