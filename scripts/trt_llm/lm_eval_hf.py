################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2025-03-13 15:20:40
# @Details  : evaluate the modelopt quantized LLM with the LM-Eval-Harness
################################################################################
"""
Adapted from https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_eval/lm_eval_hf.py
"""
from typing import Optional

from lm_eval.utils import simple_parse_args_string
from lm_eval.__main__ import cli_evaluate, parse_eval_args, setup_parser
from lm_eval.api.model import T
from lm_eval.models.huggingface import HFLM
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins import register_hf_attentions_on_the_fly
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from modelopt_quantize import (
    get_model_type,
    get_calib_dataloader,
    create_quant_cfg,
)


def create_from_arg_obj(
    cls: type[T], arg_dict: dict, additional_config: Optional[dict] = None
) -> T:
    """Overrides the HFLM.create_from_arg_obj"""

    qformat = arg_dict.pop("quantization", None)
    kv_cache_qformat = arg_dict.pop('kv_cache_qformat', None)
    awq_block_size = arg_dict.pop('awq_block_size', 128)
    auto_quantize_bits = arg_dict.pop("auto_quantize_bits", None)
    calib_batch_size = arg_dict.pop("calib_batch_size", None)
    calib_size = arg_dict.pop("calib_size", 512)

    additional_config = {} if additional_config is None else additional_config
    additional_config = {k: v for k, v in additional_config.items() if v is not None}

    model_obj = cls(**arg_dict, **additional_config)
    model_obj.tokenizer.padding_side = "left"

    if qformat:
        if not calib_batch_size:
            calib_batch_size = model_obj.batch_size

        print(f'quantization: {qformat}, kv_cache_qformat: {kv_cache_qformat}, '
              f'calib_size: {calib_size}, calib_batch_size: {calib_batch_size}')

        model_type = get_model_type(model_obj)
        device = model_obj.device
        if hasattr(model_obj, "model"):
            device = model_obj.model.device
            lm = model_obj.model
        else:
            lm = model_obj

        calib_dataloader = get_calib_dataloader(
            dataset_name="cnn_dailymail",
            tokenizer=model_obj.tokenizer,
            batch_size=calib_batch_size,
            num_samples=calib_size,
            device=device,
        )
        calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

        quant_cfg = create_quant_cfg(model_type, qformat, kv_cache_qformat, awq_block_size)

        quantize_bmm_attention = False
        for key in quant_cfg["quant_cfg"]:
            if "bmm_quantizer" in key:
                quantize_bmm_attention = True
        if quantize_bmm_attention:
            register_hf_attentions_on_the_fly(lm)

        lm = mtq.quantize(lm, quant_cfg, calibrate_loop)
        mtq.print_quant_summary(lm)
        # Fold weights for faster evaluation.
        mtq.fold_weight(lm)

    return model_obj


HFLM.create_from_arg_obj = classmethod(create_from_arg_obj)


def setup_parser_with_modelopt_args():
    parser = setup_parser()
    parser.add_argument(
        "--qformat",
        type=str,
        help='Quantization format("fp8", "int8", "int8_sq", "int4_awq", "w4a8_awq")'
    )
    parser.add_argument(
        "--kv_cache_qformat",
        type=str,
        required=False,
        default="fp8",
        choices=["fp8", "nvfp4", "none"],
        help="Specify KV cache quantization format, default to fp8 if not provided",
    )
    parser.add_argument(
        "--calib_batch_size", type=int, help="Batch size for quantization calibration"
    )
    parser.add_argument(
        "--calib_size", type=int, help="Calibration size for quantization", default=512
    )
    parser.add_argument("--awq_block_size", type=int, default=128)
    return parser


if __name__ == "__main__":
    parser = setup_parser_with_modelopt_args()
    args = parse_eval_args(parser)
    model_args = simple_parse_args_string(args.model_args)

    if args.trust_remote_code:
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        model_args["trust_remote_code"] = True
        args.trust_remote_code = None

    model_args.update(
        {
            "quantization": args.qformat,
            "kv_cache_qformat": args.kv_cache_qformat,
            "calib_batch_size": args.calib_batch_size,
            "calib_size": args.calib_size,
        }
    )

    args.model_args = model_args

    cli_evaluate(args)

# Usage:
# Baseline: CUDA_VISIBLE_DEVICES=2,3 python3 lm_eval_hf.py --model hf --model_args pretrained=<HF model>,trust_remote_code=True,parallelize=True --tasks mmlu --batch_size 8
# Quantized: CUDA_VISIBLE_DEVICES=2,3 python3 lm_eval_hf.py --model hf --model_args pretrained=<HF model>,trust_remote_code=True,parallelize=True --qformat w4a8_awq --tasks mmlu --calib_size 512 --batch_size 8
