import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def split_dataset(data, test_size=0.2):
    train_data, val_data = train_test_split(data, test_size=test_size)
    return train_data, val_data

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

def main_worker(rank, world_size):
    setup(rank, world_size)

    dataset_path = "dataset.jsonl"
    model_name = "google/mt5-small"
    batch_size = 8
    epochs = 80
    learning_rate = 5e-5
    max_input_length = 512
    max_output_length = 128

    device = torch.device(f"cuda:{rank}")

    data = load_dataset(dataset_path)
    train_data, val_data = split_dataset(data)

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model = DDP(model, device_ids=[rank])

    train_dataset = ArticleTitleDataset(train_data, tokenizer, max_input_length, max_output_length)
    val_dataset = ArticleTitleDataset(val_data, tokenizer, max_input_length, max_output_length)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train_model(model, train_dataloader, optimizer, device)
        val_loss = validate_model(model, val_dataloader, device)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")

    if rank == 0:
        model.module.save_pretrained("title_generation_model")
        tokenizer.save_pretrained("title_generation_model")

    cleanup()

def main():
    world_size = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

