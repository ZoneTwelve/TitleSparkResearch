from src.quantization.quantize_bitsandbytes import quantize_bitsandbytes
import os

def test_quantize_bitsandbytes():
    model_name = "t5-small"
    save_path = "./quantized_model_bitsandbytes"
    quantize_bitsandbytes(model_name, save_path)
    assert os.path.exists(save_path)

def test_quantize_onnx():
    model_name = "t5-small"
    save_path = "./quantized_model_onnx"
    quantize_onnx(model_name, save_path)
    assert os.path.exists(save_path)
