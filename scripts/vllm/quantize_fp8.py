################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-06-25 10:49:07
# @Details  : Apply fp8 quantization to LLMs on vLLM
################################################################################
import fire
from transformers import AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Credit to: https://github.com/vllm-project/vllm/blob/main/docs/source/quantization/fp8.rst
class LLMCompressor:
    def __init__(self,
                 model_path: str,
                 saved_path: str):
        self.saved_path = saved_path

        # Configure the simple PTQ quantization
        self.recipe = QuantizationModifier(
              targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

        self.model = SparseAutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype="auto",
            trust_remote_code=True)


    def apply_fp8(self):
        # Apply the quantization algorithm.
        oneshot(model=self.model, recipe=self.recipe)

        self.model.save_pretrained(self.saved_path)
        self.tokenizer.save_pretrained(self.saved_path)


def main(model_path: str,
         saved_path: str,
         ):
    llm_compressor = LLMCompressor(model_path, saved_path)
    llm_compressor.apply_fp8()


if __name__ == "__main__":
    fire.Fire(main)
