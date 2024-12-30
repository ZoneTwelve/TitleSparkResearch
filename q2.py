from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import torch

def quantize_mt5(model_name="google/mt5-small", save_path="quantized_mt5"):
    """
    Quantizes an mT5 model using bitsandbytes.

    Args:
        model_name: Name of the pre-trained mT5 model.
        save_path: Path to save the quantized model.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the model with 8-bit quantization using bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0, # Adjust this threshold as needed
            llm_int8_has_fp16_weight=False # Set to True if using fp16 weights
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Automatically distribute across available GPUs
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32 # Use bfloat16 if GPU is available
        )

        # Save the quantized model and tokenizer
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print(f"Quantized model saved to {save_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import fire
    fire.Fire(quantize_mt5)
