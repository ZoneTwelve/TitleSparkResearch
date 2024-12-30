from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import torch

def quantize_bitsandbytes(model_name="google/mt5-small", save_path="quantized_mt5"):
    """
    Quantize a model using BitsAndBytes for 8-bit inference.

    Args:
        model_name (str): Pretrained model name or path.
        save_path (str): Path to save the quantized model.

    Returns:
        None
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure and load model with 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  # Optional threshold
            llm_int8_has_fp16_weight=False  # Use FP16 weights if enabled
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float32
        )

        # Save the quantized model and tokenizer
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Quantized model saved to {save_path}")

    except Exception as e:
        print(f"Quantization failed: {e}")

if __name__ == "__main__":
    import fire
    fire.Fire(quantize_bitsandbytes)
