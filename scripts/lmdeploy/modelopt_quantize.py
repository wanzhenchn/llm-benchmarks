# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


"""
Adapted from https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/llm_ptq/hf_ptq.py
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Union
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import save_torch_state_dict

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

RAND_SEED = 1234
MAX_SEQ_LEN = 2048

QUANT_CFG_CHOICES = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
}

MODEL_NAME_PATTERN_MAP = {
    "GPT2": "gpt",
    "Mllama": "mllama",
    "Llama": "llama",
    "Mistral": "llama",
    "GPTJ": "gptj",
    "FalconForCausalLM": "falcon",
    "RWForCausalLM": "falcon",
    "baichuan": "baichuan",
    "MPT": "mpt",
    "Bloom": "bloom",
    "ChatGLM": "chatglm",
    "QWen": "qwen",
    "RecurrentGemma": "recurrentgemma",
    "Gemma2": "gemma2",
    "Gemma": "gemma",
    "phi3small": "phi3small",
    "phi3": "phi3",
    "PhiMoEForCausalLM": "phi3",
    "phi": "phi",
    "TLGv4ForCausalLM": "phi",
    "MixtralForCausalLM": "llama",
    "ArcticForCausalLM": "llama",
    "StarCoder": "gpt",
    "Dbrx": "dbrx",
    "T5": "t5",
    "GLM": "glm",
    "InternLM2ForCausalLM": "internlm",
    "ExaoneForCausalLM": "exaone",
    "Nemotron": "gpt",
}

mto.enable_huggingface_checkpointing()


def get_model_type(model):
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN):
    print(f"Initializing tokenizer from {ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=True,
    )

    if "qwen" in type(tokenizer).__name__.lower():
        # qwen use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None, f"Pad token for {ckpt_path} cannot be set!"

    return tokenizer


def get_model(ckpt_path, device="cuda", gpu_mem_percentage=0.8):
    print(f"Initializing model from {ckpt_path}")

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    # Note: Forcibly converting the model precision between bf16 and fp16 may introduce accuracy drop
    model_kwargs = {"torch_dtype": "auto"}

    hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)

    if hf_config.model_type == "llava":
        from transformers import LlavaForConditionalGeneration

        hf_llava = LlavaForConditionalGeneration.from_pretrained(
            ckpt_path, device_map=device_map, **model_kwargs
        )
        model = hf_llava.language_model
    else:
        from accelerate import infer_auto_device_map, init_empty_weights
        from accelerate.utils import get_max_memory

        with init_empty_weights():
            # When computing the device_map, assuming half precision by default,
            # unless specified by the hf_config.
            torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
            model = AutoModelForCausalLM.from_config(
                hf_config, torch_dtype=torch_dtype, trust_remote_code=True
            )
        max_memory = get_max_memory()
        inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)
        on_cpu = "cpu" in inferred_device_map.values()

        if on_cpu:
            for device in max_memory.keys():
                if isinstance(device, int):
                    max_memory[device] *= gpu_mem_percentage
            print(
                "Model does not fit to the GPU mem. "
                f"We apply the following memmory limit for calibration: \n{max_memory}\n"
                "If you hit GPU OOM issue, please adjust `gpu_mem_percentage` or "
                "reduce the calibration `batch_size` manually."
            )

        model = AutoModelForCausalLM.from_pretrained(ckpt_path,
                                                     device_map=device_map,
                                                     **model_kwargs,
                                                     trust_remote_code=True)
    model.eval()
    if device == "cuda":
        if not all("cuda" in str(param.device) for param in model.parameters()):
            print("Warning: Some parameters are not on a GPU. Calibration can be slow or hit OOM")

    return model


def get_calib_dataloader(data="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         max_sample_length=512,
                         device=None):
    print(f"Loading calibration dataset {data}")
    if data == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train")
        dataset = dataset["text"][:calib_size]
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        dataset = dataset["article"][:calib_size]
    elif data == "ultrachat_2k":
        dataset = load_dataset("mgoin/ultrachat_2k", split="train_sft")
        dataset = dataset["prompt"][:calib_size]
    else:
        raise NotImplementedError

    batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                max_length=max_sample_length)
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]

    calib_dataloader = DataLoader(batch_encoded,
                                  batch_size=batch_size,
                                  shuffle=False)
    return calib_dataloader


def quantize_model(args, model, quant_cfg, calib_dataloader=None):

    def calibrate_loop(model):
        if calib_dataloader is None:
            return
        """Adjusts weights and scaling factors based on selected algorithms."""
        with torch.no_grad():
            for _, data in enumerate(tqdm(calib_dataloader, desc=f"Applying {args.qformat}")):
                model(data)

    print("Starting quantization...")
    start_time = time.time()
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization done. Total time used: {end_time - start_time}s")
    return model


class Export2Onellm:
    def __init__(self,
                 args,
                 model: nn.Module,
                 tokenizer: nn.Module):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.export_path = args.output_dir

        assert self._is_fp8(model), "Only supports FP8 OneLLM export."

    @staticmethod
    def _is_fp8(model):
        for _, layer in model.named_modules():
            if model == layer:
                continue

            if isinstance(layer, nn.Module):
                if "TensorQuantizer" in type(layer).__name__ and layer.is_enabled:
                    return layer.num_bits == (4, 3)

                return_value = Export2Onellm._is_fp8(layer)
                if return_value is not None:
                    return return_value
        return None

    @staticmethod
    def _convert_scales(key, value):
        """Replaces the names of *quantizer._amax to _scale."""
        replacements = {
            "weight_quantizer._amax": "weight_scale",
            "input_quantizer._amax": "input_scale",
            "k_proj.output_quantizer._amax": "k_scale",
            "v_proj.output_quantizer._amax": "v_scale",
        }
        for old_suffix, new_suffix in replacements.items():
            if key.endswith(old_suffix):
                new_key = key[: len(key) - len(old_suffix)] + new_suffix
                new_value = (value / 448).to(torch.float32)
                return new_key, new_value

        return key, value

    @staticmethod
    def _convert_weights_to_fp8(state_dict: Dict[str, torch.tensor],
                                weights_to_convert: List[str],
                                ) -> Dict[str, torch.tensor]:
        """Converts the original weights to FP8E4M3 from FP16."""
        for weight_name in weights_to_convert:
            weight_scale_name = weight_name + "_scale"
            if weight_scale_name not in state_dict.keys():
                continue
            loaded_weight = state_dict.pop(weight_name)
            scale = state_dict[weight_scale_name]
            qweight_name = weight_name.replace('weight', 'qweight')
            state_dict[qweight_name] = (loaded_weight.cpu() / scale.cpu()).to(torch.float8_e4m3fn)
        return state_dict

    def convert_to_onellm_compatible_weights(self,
                                             input_state_dict: Dict[str, torch.tensor]):
        """Util function to modify the modelopt state dict to OneLLM checkpoint."""
        weights_to_convert = []
        onellm_state_dict = {}
        for key, value in input_state_dict.items():
            if key.endswith("_amax"):
                new_key, new_value = Export2Onellm._convert_scales(key, value)
                # Only add if the replacement happened.
                if key != new_key:
                    onellm_state_dict[new_key] = new_value
            else:
                weights_to_convert.append(key)
                onellm_state_dict[key] = value
        # Conversion can only happen after all the amax values are read.
        onellm_state_dict = Export2Onellm._convert_weights_to_fp8(
            onellm_state_dict, weights_to_convert)
        onellm_state_dict = OrderedDict(
            sorted(onellm_state_dict.items(),
                   key=lambda x: (
                       int(x[0].split(".")[2]) if "model.layers." in x[0] else float('inf'), x[0])
                   )
        )
        return onellm_state_dict

    def export(self):
        onellm_state_dict = self.convert_to_onellm_compatible_weights(
            self.model.state_dict())

        # create directory
        Path(self.export_path).mkdir(parents=True, exist_ok=True)

        # save the state dict
        save_torch_state_dict(onellm_state_dict, self.export_path)

        # save the config.json
        config_path = Path(self.export_path, "config.json")
        setattr(self.model.config,
                "quantization_config", {
                    "quant_method": self.args.qformat,
                    "activation_scheme": "static",
                    "kv_cache_scheme": 'fp8' if 'int8_sq' not in self.args.qformat else None,
                    "version": "gemm"
                }
        )
        self.model.config.to_json_file(config_path)

        # save the tokenizer
        self.tokenizer.save_pretrained(Path(self.export_path))


def main(args):
    if not torch.cuda.is_available():
        raise OSError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    model = get_model(args.model_dir, args.device)
    tokenizer = get_tokenizer(args.model_dir)
    model_type = get_model_type(model)

    device = model.device
    if hasattr(model, "model"):
        device = model.model.device

    if args.qformat in ["full_prec", "int8_wo", "int4_wo"
                        ] and args.kv_cache_dtype is None:
        print(f"No quantization applied, export {args.dtype} model")
    else:
        calib_dataloader = get_calib_dataloader(
            data=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            calib_size=args.calib_size,
            device=args.device,
        )

        if args.qformat in QUANT_CFG_CHOICES:
            quant_cfg = QUANT_CFG_CHOICES[args.qformat]
        else:
            raise ValueError(f"Unsupported quantization format: {args.qformat}")

        if "awq" in args.qformat:
            quant_cfg = copy.deepcopy(getattr(mtq, QUANT_CFG_CHOICES[args.qformat]))
            weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]  # type: ignore
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            if args.awq_block_size:
                weight_quantizer["block_sizes"][-1] = args.awq_block_size

        # Always turn on FP8 kv cache to save memory footprint.
        # For int8_sq, we do not quantize kv cache to preserve accuracy.
        enable_quant_kv_cache = "int8_sq" not in args.qformat
        quant_cfg["quant_cfg"]["*output_quantizer"] = {
            "num_bits": 8 if args.qformat == "int8_sq" else (4, 3),
            "axis": None,
            "enable": enable_quant_kv_cache,
        }
        print(json.dumps(quant_cfg, indent=4))

        # Only run single sample for preview
        input_ids = next(iter(calib_dataloader))[0:1]
        generated_ids_before_ptq = model.generate(input_ids, max_new_tokens=100)

        model = quantize_model(args, model, quant_cfg, calib_dataloader)
        # Lets print the quantization summary
        mtq.print_quant_summary(model)

        # Run some samples
        generated_ids_after_ptq = model.generate(input_ids, max_new_tokens=100)
        print(f"\nexample test input: {tokenizer.batch_decode(input_ids)}\n"
              f"outputs before {args.qformat}: {tokenizer.batch_decode(generated_ids_before_ptq[:, input_ids.shape[1]:])}\n"
              f"outputs after {args.qformat} : {tokenizer.batch_decode(generated_ids_after_ptq[:, input_ids.shape[1]:])}\n")

    with torch.inference_mode():
        if model_type is None:
            print(f"Unknown model type {type(model).__name__}. Continue exporting...")
            model_type = f"unknown:{type(model).__name__}"

        start_time = time.time()
        export_helper = Export2Onellm(args, model, tokenizer)
        export_helper.export()
        end_time = time.time()
        print(f"Quantized model exported to: {args.output_dir}. "
              f"Total time used {end_time - start_time}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",
                        help="Specify where the HuggingFace model is",
                        required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--qformat",
        help="Quantization format.",
        default="fp8",
        choices=["fp8", "int8", "int8_sq", "int4_awq", "w4a8_awq"]
    )
    parser.add_argument("--batch-size",
                        help="Batch size for calibration.",
                        type=int,
                        default=1)
    parser.add_argument(
        "--dataset",
        help="dataset name for quantization calibration.",
        default="cnn_dailymail",
        choices=["cnn_dailymail", "ultrachat_2k", "pileval"]
    )
    parser.add_argument("--calib-size",
                        help="Number of samples for calibration.",
                        type=int,
                        default=512)
    parser.add_argument("--output-dir", default="exported_model")
    parser.add_argument("--awq-block-size", type=int, default=128)
    args = parser.parse_args()

    main(args)
