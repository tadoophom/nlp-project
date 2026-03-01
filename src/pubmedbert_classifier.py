"""Transformer-based protein-disease relation classifier."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch


LABEL_NAMES = ["associated", "not_associated", "incidental"]

LABEL_TO_POLARITY = {
    "associated": "Positive",
    "not_associated": "Negative",
    "incidental": "Neutral",
}

DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent
    / "models"
    / "hfpef_quality"
    / "2026-02-02"
    / "scibert_v4"
    / "final"
)


class PubMedBERTClassifier:
    """Classifier using a fine-tuned transformer for relation extraction."""

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None, max_length: int = 256):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._loaded:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please run the fine-tuning script first."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.eval()

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)

        self._loaded = True

    def classify(self, sentence: str) -> dict:
        """
        Classify a sentence for protein-disease relation.

        Returns:
            dict with keys:
                - label: one of associated, not_associated, incidental
                - polarity: Positive, Negative, or Neutral (for compatibility)
                - confidence: probability score
                - probabilities: dict of all class probabilities
        """
        self._load_model()

        inputs = self._tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()

        label = LABEL_NAMES[pred_id]
        polarity = LABEL_TO_POLARITY[label]

        return {
            "label": label,
            "polarity": polarity,
            "confidence": confidence,
            "probabilities": {
                name: probs[i].item() for i, name in enumerate(LABEL_NAMES)
            },
        }

    def classify_batch(self, sentences: list[str], batch_size: int = 32) -> list[dict]:
        """Classify multiple sentences efficiently."""
        self._load_model()

        results = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]

            inputs = self._tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_ids = torch.argmax(probs, dim=-1)

            for j in range(len(batch)):
                pred_id = pred_ids[j].item()
                confidence = probs[j, pred_id].item()
                label = LABEL_NAMES[pred_id]
                polarity = LABEL_TO_POLARITY[label]

                results.append({
                    "label": label,
                    "polarity": polarity,
                    "confidence": confidence,
                    "probabilities": {
                        name: probs[j, k].item() for k, name in enumerate(LABEL_NAMES)
                    },
                })

        return results


@lru_cache(maxsize=1)
def get_classifier(model_path: Optional[str] = None) -> PubMedBERTClassifier:
    """Get a cached classifier instance."""
    path = Path(model_path) if model_path else None
    return PubMedBERTClassifier(model_path=path)


def is_model_available(model_path: Optional[Path] = None) -> bool:
    """Check if the fine-tuned model is available."""
    path = model_path or DEFAULT_MODEL_PATH
    return path.exists() and (path / "config.json").exists()
