################################################################################
# @Author   : wanzhenchn@gmail.com
# @Date     : 2024-04-30 10:49:07
# @Details  : Apply AWQ-W4A16 quantization to LLMs on VLLM
################################################################################
import fire
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


class AWQ:
    def __init__(self,
                 model_path: str,
                 saved_path: str):
        self.saved_path = saved_path

        self.quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
             "version": "GEMM"
        }

        # Load model
        self.model = AutoAWQForCausalLM.from_pretrained(
            model_path, **{"low_cpu_mem_usage": True}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def apply_awq(self):
       # Quantize
       self.model.quantize(self.tokenizer, quant_config=self.quant_config)

       # Save quantized model
       self.model.save_quantized(self.saved_path)
       self.tokenizer.save_pretrained(self.saved_path)


def main(model_path: str,
         saved_path: str
         ):
    awq_helper = AWQ(model_path, saved_path)
    awq_helper.apply_awq()


if __name__ == "__main__":
    fire.Fire(main)
