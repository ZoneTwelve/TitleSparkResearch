import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_title(article, model_path="title_generation_model", max_input_length=512, max_output_length=128, device=None):
    """
    Generate a title for the given article using a pretrained T5 model.

    Args:
        article (str): Input article text.
        model_path (str): Path to the pretrained model.
        max_input_length (int): Maximum token length for the input.
        max_output_length (int): Maximum token length for the output.
        device (torch.device, optional): Device to run the model on.

    Returns:
        str: Generated title.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    # Tokenize input
    input_encoding = tokenizer(
        article,
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
        return_tensors="pt"
    )

    input_ids = input_encoding["input_ids"].to(device)
    attention_mask = input_encoding["attention_mask"].to(device)

    # Generate the title
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_output_length,
            num_beams=4,  # Beam search for better results
            early_stopping=True
        )

    # Decode and return the generated title
    title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return title

if __name__ == "__main__":
    # Example usage
    example_article = (
        "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed "
        "to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits "
        "associated with a human mind such as learning and problem-solving."
    )

    # Set model path and device
    model_path = "title_generation_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate title
    generated_title = generate_title(example_article, model_path=model_path, device=device)
    print("Generated Title:", generated_title)
