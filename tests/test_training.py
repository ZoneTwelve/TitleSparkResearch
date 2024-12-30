import pytest
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.train.train_torch import ArticleTitleDataset, train_model, validate_model

@pytest.fixture
def sample_data():
    # Sample dataset for testing
    return [
        {"article": "This is a sample article.", "title": "Sample Title"},
        {"article": "Another article for testing.", "title": "Test Title"},
    ]

@pytest.fixture
def tokenizer():
    return T5Tokenizer.from_pretrained("t5-small")

@pytest.fixture
def model():
    return T5ForConditionalGeneration.from_pretrained("t5-small")

def test_dataset_initialization(sample_data, tokenizer):
    dataset = ArticleTitleDataset(sample_data, tokenizer)
    assert len(dataset) == 2
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample

def test_training_step(sample_data, tokenizer, model):
    from torch.utils.data import DataLoader
    import torch.optim as optim

    dataset = ArticleTitleDataset(sample_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    device = "cpu"

    loss = train_model(model, dataloader, optimizer, device)
    assert loss > 0  # Loss should be a positive value

def test_validation_step(sample_data, tokenizer, model):
    from torch.utils.data import DataLoader

    dataset = ArticleTitleDataset(sample_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)
    device = "cpu"

    loss = validate_model(model, dataloader, device)
    assert loss > 0  # Loss should be a positive value
