################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-06-25 10:49:07
# @Details  : Apply fp8 quantization to LLMs on vLLM
################################################################################
import fire
from datasets import load_dataset
from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

# Credit to: https://github.com/vllm-project/vllm/blob/main/docs/source/quantization/fp8.rst
class AutoFP8:
    def __init__(self,
                 model_path: str,
                 saved_path: str,
                 calib_size: int = 512,
                 activation_scheme: str = "static"):
        self.saved_path = saved_path
        self.calib_size = calib_size

        self.quantize_config = BaseQuantizeConfig(
            quant_method="fp8", activation_scheme=activation_scheme)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoFP8ForCausalLM.from_pretrained(model_path,
                                                        self.quantize_config)


    def apply_fp8(self):
        # Load and tokenize 512 dataset samples for calibration of activation scales
        ds = load_dataset("mgoin/ultrachat_2k", split="train_sft").select(
            range(self.calib_size))
        examples = [self.tokenizer.apply_chat_template(
            batch["messages"], tokenize=False) for batch in ds]
        examples = self.tokenizer(examples, padding=True, truncation=True,
                                  return_tensors="pt").to("cuda")

        # quantize, and save checkpoint
        self.model.quantize(examples)
        self.model.save_quantized(self.saved_path)


def main(model_path: str,
         saved_path: str,
         calib_size: int = 512,
         ):
    fp8_helper = AutoFP8(model_path, saved_path, calib_size)
    fp8_helper.apply_fp8()


if __name__ == "__main__":
    fire.Fire(main)
