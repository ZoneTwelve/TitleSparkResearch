import pytest
from src.inference.preview_tokenization import preview_tokenization

@pytest.fixture
def mock_dataset(tmp_path):
    file_path = tmp_path / "mock_dataset.jsonl"
    with open(file_path, "w") as f:
        f.write('{"article": "This is an article.", "title": "Title"}\n')
        f.write('{"article": "Another article.", "title": "Another Title"}\n')
    return str(file_path)

def test_preview_tokenization(mock_dataset):
    model_name = "t5-small"
    preview_tokenization(mock_dataset, model_name=model_name, sample_size=1)
