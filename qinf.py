from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def inference(model_path="quantized_mt5", input_text="translate English to German: Hello, how are you?"):
    """
    Performs inference with a quantized mT5 model.

    Args:
        model_path: Path to the quantized model.
        input_text: The input text for translation.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device) # Move input to correct device
        
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model.generate(input_ids)

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {input_text}")
        print(f"Output: {decoded_output}")

    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    import fire
    fire.Fire(inference)
