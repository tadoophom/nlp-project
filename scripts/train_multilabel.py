"""
Multi-label classification training.

Allows sentences to have multiple labels simultaneously,
e.g., both positive AND negative aspects in one sentence.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


LABELS = ['positive', 'negative', 'no_association']


class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


def convert_to_multilabel(label_str):
    """Convert single label to multi-label binary vector."""
    vec = [0.0, 0.0, 0.0]
    idx = LABELS.index(label_str)
    vec[idx] = 1.0
    return vec


def train_multilabel(
    data_path: Path,
    output_path: Path,
    model_name: str = "allenai/scibert_scivocab_uncased",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    threshold: float = 0.5
):
    # Load data
    with open(data_path) as f:
        data = json.load(f)
    
    texts = [d['sentence'] for d in data]
    labels = [convert_to_multilabel(d['label']) for d in data]
    
    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42
    )
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        problem_type="multi_label_classification"
    )
    
    # Datasets
    train_ds = MultiLabelDataset(train_texts, train_labels, tokenizer)
    val_ds = MultiLabelDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits)
                preds = (probs > threshold).int()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
        
        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        for i, label in enumerate(LABELS):
            acc = (all_preds[:, i] == all_labels[:, i]).mean()
            print(f"  {label}: {acc:.1%}")
    
    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/labeled.json")
    parser.add_argument("--output", default="models/scibert-multilabel")
    parser.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    
    train_multilabel(
        data_path=Path(args.data),
        output_path=Path(args.output),
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr
    )
