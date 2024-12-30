# TitleSparkResearch

ZoneTwelve TitleSparkResearch is a Python-based project that leverages transformer models for title generation. It supports training, inference, quantization, and dataset preprocessing for encoder-decoder architectures like T5 and mT5.

## Features

- **Training**: Train transformer models using PyTorch or TensorFlow on single or multiple GPUs.
- **Inference**: Generate titles for articles with pretrained or fine-tuned models.
- **Quantization**: Optimize models using `bitsandbytes` or ONNX for efficient inference.
- **Utilities**: Tools for dataset preprocessing, tokenization previews, and more.
- **Testing**: Comprehensive test scripts to ensure functionality and reliability.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/zonetwelve-TitleSparkResearch.git
   cd zonetwelve-TitleSparkResearch
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Optionally, install the package:
   ```bash
   pip install .
   ```

---

## Usage

### Training
To train a model:
```bash
python -m src.train.train_torch --dataset_path path/to/dataset.jsonl --model_name google/mt5-small
```

### Inference
Generate a title for an article:
```bash
python -m src.inference.infer_torch --article "This is a sample article about AI and machine learning."
```

### Quantization
Quantize the model using `bitsandbytes`:
```bash
python -m src.quantization.quantize_bitsandbytes --model_name google/mt5-small --save_path quantized_mt5
```

Quantize the model using ONNX:
```bash
python -m src.quantization.quantize_onnx --model_name google/mt5-small --save_path quantized_mt5_onnx
```

### Tokenization Preview
Preview tokenization on a sample dataset:
```bash
python -m src.inference.preview_tokenization --dataset_path path/to/dataset.jsonl --model_name google/mt5-small
```

---

## Project Structure

```
zonetwelve-TitleSparkResearch/
├── src/
│   ├── train/               # Training scripts
│   ├── inference/           # Inference and tokenization scripts
│   ├── quantization/        # Quantization scripts
│   ├── utils/               # Utility scripts for dataset handling
├── tests/                   # Test scripts for all modules
├── LICENSE                  # License file (Apache 2.0)
├── README.md                # Project documentation
├── setup.py                 # For traditional installation
└── pyproject.toml           # For modern packaging and build
```

---

## Dataset Format

The dataset should be in JSONL format with each line containing an object with `article` and `title` keys:
```json
{"article": "This is an article about AI.", "title": "AI Explained"}
{"article": "This is another article about ML.", "title": "Understanding ML"}
```

---

## Dependencies

- Python >= 3.8
- PyTorch >= 1.13.0
- Transformers >= 4.33.0
- Optimum (for ONNX quantization)
- scikit-learn (for data splitting)
- pytest (for testing)

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

## Testing

Run all tests using `pytest`:
```bash
pytest tests/
```

---

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
