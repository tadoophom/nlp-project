"""
Domain expansion via transfer learning.

Adapts a trained model (e.g., HFpEF) to a new disease domain (e.g., HFrEF, CKD)
with minimal new labeled data.

Strategy:
1. Load pre-trained model
2. Freeze base layers, train classifier head on new domain
3. Unfreeze all layers, fine-tune with very low learning rate
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split


LABELS = ['positive', 'negative', 'no_association']


class TextDataset(Dataset):
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
            'labels': torch.tensor(LABELS.index(self.labels[idx]))
        }


def freeze_base_layers(model):
    """Freeze all layers except classifier head."""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


def transfer_to_domain(
    source_model: str,
    target_data: Path,
    output_path: Path,
    phase1_epochs: int = 3,
    phase2_epochs: int = 2,
    phase1_lr: float = 1e-4,
    phase2_lr: float = 1e-6,
    batch_size: int = 8
):
    """
    Two-phase transfer learning.
    
    Phase 1: Train classifier head only (fast adaptation)
    Phase 2: Fine-tune all layers with low LR (refinement)
    """
    # Load data
    with open(target_data) as f:
        data = json.load(f)
    
    texts = [d['sentence'] for d in data]
    labels = [d['label'] for d in data]
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"Target domain: {len(train_texts)} train, {len(val_texts)} val")
    
    # Load pre-trained model
    print(f"Loading source model: {source_model}")
    tokenizer = AutoTokenizer.from_pretrained(source_model)
    model = AutoModelForSequenceClassification.from_pretrained(source_model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Datasets
    train_ds = TextDataset(train_texts, train_labels, tokenizer)
    val_ds = TextDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    def evaluate():
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        return correct / total
    
    def train_epoch(optimizer):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        
        return total_loss / len(train_loader)
    
    # Phase 1: Classifier head only
    print(f"\n=== Phase 1: Classifier Head Only ===")
    freeze_base_layers(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=phase1_lr)
    
    for epoch in range(phase1_epochs):
        loss = train_epoch(optimizer)
        acc = evaluate()
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, val_acc={acc:.1%}")
    
    # Phase 2: Full fine-tuning
    print(f"\n=== Phase 2: Full Fine-tuning ===")
    unfreeze_all(model)
    optimizer = AdamW(model.parameters(), lr=phase2_lr)
    
    for epoch in range(phase2_epochs):
        loss = train_epoch(optimizer)
        acc = evaluate()
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, val_acc={acc:.1%}")
    
    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\nSaved transferred model to {output_path}")
    print(f"Final validation accuracy: {evaluate():.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer model to new domain")
    parser.add_argument("--source", required=True, help="Source model path")
    parser.add_argument("--data", required=True, help="Target domain labeled data (JSON)")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--phase1-epochs", type=int, default=3)
    parser.add_argument("--phase2-epochs", type=int, default=2)
    args = parser.parse_args()
    
    transfer_to_domain(
        source_model=args.source,
        target_data=Path(args.data),
        output_path=Path(args.output),
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs
    )
