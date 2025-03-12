import fire
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def main(model_path: str, output_path: str):

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer,
                   quant_config=quant_config,
                   calib_data="pileval",
                   max_calib_samples=128
                   )

    # Save quantized model
#    output_path= model_path.split("/")[-1] + "-awq-w4a16-hf"
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    print(f'Model is quantized and saved at "{output_path}"')


if __name__ == "__main__":
    fire.Fire(main)
