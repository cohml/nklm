import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        return inputs.input_ids.squeeze(), inputs.attention_mask.squeeze()

# Example training texts
train_texts = [
    "I love to go to the park.",
    "The cat is playing with a ball.",
    "She sings beautifully.",
    "He is a good dancer."
]

# Create the dataset and data loader
train_dataset = CustomDataset(train_texts)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
