"""PubMedBERT-based protein-disease relation classifier."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset

RelationLabel = Literal["associated", "not_associated", "incidental"]
LABEL2ID: Dict[str, int] = {"associated": 0, "not_associated": 1, "incidental": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# Section headers that confuse the model
SECTION_HEADERS = re.compile(
    r'^(INTRODUCTION|BACKGROUND|METHODS|RESULTS|DISCUSSION|CONCLUSION|OBJECTIVE|PURPOSE|AIM|AIMS):?\s+',
    re.IGNORECASE
)


def preprocess_sentence(sentence: str) -> str:
    """Strip section headers that confuse classification."""
    return SECTION_HEADERS.sub('', sentence).strip()

MODEL_NAME = "allenai/scibert_scivocab_uncased"  # Best performer in CV comparison

# Available biomedical models (ranked by typical performance)
BIOMEDICAL_MODELS = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "biobert": "dmis-lab/biobert-v1.1",
    "scibert": "allenai/scibert_scivocab_uncased",
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "biomedbert-large": "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
}


def get_model_name(model_key: str) -> str:
    """Resolve model key to full HuggingFace model name."""
    return BIOMEDICAL_MODELS.get(model_key, model_key)


class RelationDataset(Dataset):
    """Dataset for protein-disease relation classification."""

    def __init__(
        self,
        sentences: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 256,
    ):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class PubMedBERTClassifier:
    """Wrapper for PubMedBERT relation classification."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_name = model_path or MODEL_NAME

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentence: str) -> Tuple[str, float]:
        """Predict relation label and confidence for a single sentence."""
        sentence = preprocess_sentence(sentence)
        inputs = self.tokenizer(
            sentence,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = probs.argmax(dim=-1).item()
            confidence = probs[0, pred_id].item()

        return ID2LABEL[pred_id], confidence

    def predict_batch(
        self, sentences: List[str], batch_size: int = 16
    ) -> List[Tuple[str, float]]:
        """Predict labels for multiple sentences."""
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = [preprocess_sentence(s) for s in sentences[i : i + batch_size]]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_ids = probs.argmax(dim=-1).tolist()
                confidences = probs.max(dim=-1).values.tolist()

            for pred_id, conf in zip(pred_ids, confidences):
                results.append((ID2LABEL[pred_id], conf))

        return results

    def save(self, path: str) -> None:
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None) -> "PubMedBERTClassifier":
        """Load a fine-tuned model."""
        return cls(model_path=path, device=device)


def load_labeled_data(path: Path) -> Tuple[List[str], List[int]]:
    """Load labeled sentences from JSON file."""
    with open(path) as f:
        data = json.load(f)

    sentences = []
    labels = []
    for item in data:
        sentences.append(item["sentence"])
        labels.append(LABEL2ID[item["label"]])

    return sentences, labels


def save_labeled_data(
    sentences: List[str], labels: List[str], path: Path
) -> None:
    """Save labeled sentences to JSON file."""
    data = [{"sentence": s, "label": l} for s, l in zip(sentences, labels)]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
