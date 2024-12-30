import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def infer_quantized(model_path="quantized_mt5", input_text="translate English to German: Hello, how are you?"):
    """
    Perform inference with a quantized mT5 model.

    Args:
        model_path (str): Path to the quantized model.
        input_text (str): Input text for translation or generation.

    Returns:
        str: Generated text.
    """
    try:
        # Load tokenizer and quantized model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # Tokenize input text
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

        # Perform inference
        with torch.no_grad():
            outputs = model.generate(input_ids)

        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return None

if __name__ == "__main__":
    import fire
    fire.Fire(infer_quantized)
