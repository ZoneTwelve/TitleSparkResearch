import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split

class ArticleTitleDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_output_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]["article"]
        title = self.data[idx]["title"]

        # Tokenize input (article) and output (title)
        input_encoding = self.tokenizer(
            article, truncation=True, padding="max_length", max_length=self.max_input_length, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            title, truncation=True, padding="max_length", max_length=self.max_output_length, return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
        }

# Load the dataset
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

# Train-Test Split
def split_dataset(data, test_size=0.2):
    train_data, val_data = train_test_split(data, test_size=test_size)
    return train_data, val_data

# Training Function
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

# Validation Function
def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Main Training Script
def main():
    # Paths and configurations
    dataset_path = "dataset.jsonl"
    #model_name = "t5-small"
    model_name = "google/mt5-small"
    batch_size = 24
    epochs = 24
    learning_rate = 5e-5
    max_input_length = 512
    max_output_length = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Load dataset")
    # Load and split dataset
    data = load_dataset(dataset_path)
    train_data, val_data = split_dataset(data)

    print("Load model")
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    print("Prepare datasets")
    # Prepare datasets and dataloaders
    train_dataset = ArticleTitleDataset(train_data, tokenizer, max_input_length, max_output_length)
    val_dataset = ArticleTitleDataset(val_data, tokenizer, max_input_length, max_output_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    print("Optmizer")
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Ready to train")
    # Training loop
    for epoch in range(epochs):
        train_loss = train_model(model, train_dataloader, optimizer, device)
        val_loss = validate_model(model, val_dataloader, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

    # Save the trained model
    model.save_pretrained("title_generation_model")
    tokenizer.save_pretrained("title_generation_model")

if __name__ == "__main__":
    main()

