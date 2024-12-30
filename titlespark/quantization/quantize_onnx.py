from transformers import MT5ForConditionalGeneration, AutoTokenizer
from optimum.onnxruntime import ORTQuantizer

def quantize_onnx(model_name="google/mt5-small", save_path="quantized_mt5_onnx"):
    """
    Quantize a model using ONNX Runtime for optimized inference.

    Args:
        model_name (str): Pretrained model name or path.
        save_path (str): Path to save the ONNX quantized model.

    Returns:
        None
    """
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MT5ForConditionalGeneration.from_pretrained(model_name)

        # Initialize ONNX quantizer
        quantizer = ORTQuantizer.from_pretrained(model_name)

        # Perform quantization
        quantizer.quantize()

        # Save the quantized model
        quantized_model_path = save_path
        quantizer.save_pretrained(quantized_model_path)
        tokenizer.save_pretrained(quantized_model_path)
        print(f"ONNX quantized model saved to {quantized_model_path}")

    except Exception as e:
        print(f"Quantization failed: {e}")

if __name__ == "__main__":
    import fire
    fire.Fire(quantize_onnx)
