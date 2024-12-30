import pytest
from src.inference.infer_torch import generate_title

@pytest.fixture
def sample_article():
    return "This is a test article about AI and machine learning."

def test_inference(sample_article):
    model_path = "t5-small"
    generated_title = generate_title(sample_article, model_path=model_path, device="cpu")
    assert isinstance(generated_title, str)
    assert len(generated_title) > 0
