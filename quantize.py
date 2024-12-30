from transformers import MT5ForConditionalGeneration, AutoTokenizer
from optimum.onnxruntime import ORTQuantizer

def quantize(model_name = "google/mt5-small", save_path = "quantized_mt5"):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = MT5ForConditionalGeneration.from_pretrained(model_name)


	quantizer = ORTQuantizer.from_pretrained(model_name)
	quantizer.quantize()

	quantized_model_path = save_path 
	quantizer.save_pretrained(quantized_model_path)

if __name__ == "__main__":
	import fire
	fire.Fire(quantize)
