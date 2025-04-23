################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-06-25 10:49:07
# @Details  : Apply quantization on LLMs with llm-compressor
################################################################################
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# Credit to: https://github.com/vllm-project/llm-compressor/tree/main/examples
class LLMCompressor:
    def __init__(self,
                 model_path: str,
                 quant_method: str,
                 saved_path: str):
        self.saved_path = saved_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype="auto",
            trust_remote_code=True)

        model_type = self.model.config.model_type
        ignore_layers = ["lm_head"]

        if ("moe" in model_type) or ("deepseek" in model_type):
            ignore_layers.extend(["re:.*mlp.gate"])
        elif "mixtral" in model_type:
            ignore_layers.extend(["re:.*block_sparse_moe.gate"])
        elif "qwen2" in model_type:
            ignore_layers.extend(["re:.*mlp.shared_expert_gate"])

        # Configure the simple PTQ quantization
        if quant_method == "fp8":
            self.recipe = QuantizationModifier(
                targets="Linear", scheme="FP8", ignore=ignore_layers
            )
        elif quant_method == "awq-w4a16":
            self.recipe = [
                AWQModifier(bits=4, symmetric=False),
                QuantizationModifier(
                    ignore=ignore_layers,
                    config_groups={
                        "group_0": QuantizationScheme(
                            targets=["Linear"],
                            weights=QuantizationArgs(
                                num_bits=4,
                                type=QuantizationType.INT,
                                dynamic=False,
                                symmetric=False,
                                strategy=QuantizationStrategy.GROUP,
                                group_size=128,
                            ),
                        )
                    },
                ),
            ]
        elif quant_method == "awq-w4a8":
            self.recipe = [
                AWQModifier(bits=4, symmetric=False),
                QuantizationModifier(
                    ignore=ignore_layers,
                    config_groups={
                        "group_0": QuantizationScheme(
                            targets=["Linear"],
                            weights=QuantizationArgs(
                                num_bits=4,
                                type=QuantizationType.INT,
                                dynamic=False,
                                symmetric=False,
                                strategy=QuantizationStrategy.GROUP,
                                group_size=128,
                            ),
                            input_activations=QuantizationArgs(
                                num_bits=8,
                                type=QuantizationType.FLOAT,
                                strategy=QuantizationStrategy.TENSOR,
                                dynamic=False,
                                symmetric=True,
                            ),
                        ),
                    },
                ),
            ]


    def apply_quantize(self):
        # Apply the quantization algorithm.
        oneshot(model=self.model,
                tokenizer=self.tokenizer,
                dataset="open_platypus",
                recipe=self.recipe,
                max_seq_length=2048,
                num_calibration_samples=512,
                save_compressed=True,
                trust_remote_code_model=True,
                output_dir=self.saved_path,
                )

#        self.model.save_pretrained(self.saved_path)
        self.tokenizer.save_pretrained(self.saved_path)


def main(model_path: str,
         quant_method: str,
         saved_path: str,
         ):
    llm_compressor = LLMCompressor(model_path, quant_method, saved_path)
    llm_compressor.apply_quantize()


if __name__ == "__main__":
    fire.Fire(main)
