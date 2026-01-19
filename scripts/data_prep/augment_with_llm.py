"""Data augmentation using paraphrasing and synonym replacement."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from collections import Counter


def augment_synonym_replacement(sentence: str, n_replacements: int = 2) -> str:
    """Replace non-key words with synonyms."""
    # Words to preserve (biomedical terms, HFpEF, protein names)
    preserve_patterns = [
        r'\bHFpEF\b', r'\bHFrEF\b', r'\bheart failure\b',
        r'\bejection fraction\b', r'\b[A-Z][A-Z0-9]{2,}\b',  # Protein codes
        r'\bp\s*[<>=]\s*[\d.]+\b',  # p-values
        r'\b\d+\.?\d*\s*%\b',  # percentages
    ]
    
    # Simple synonym map for common words
    synonyms = {
        "associated": ["linked", "connected", "related", "correlated"],
        "significant": ["notable", "substantial", "meaningful", "considerable"],
        "patients": ["individuals", "subjects", "participants"],
        "study": ["research", "investigation", "analysis", "trial"],
        "showed": ["demonstrated", "revealed", "indicated", "exhibited"],
        "increased": ["elevated", "higher", "raised", "enhanced"],
        "decreased": ["reduced", "lower", "diminished", "declined"],
        "observed": ["noted", "found", "detected", "identified"],
        "suggests": ["indicates", "implies", "proposes"],
        "compared": ["relative", "versus", "contrasted"],
        "effect": ["impact", "influence", "outcome"],
        "treatment": ["therapy", "intervention", "management"],
        "levels": ["concentrations", "amounts", "values"],
        "risk": ["likelihood", "probability", "chance"],
        "outcomes": ["results", "endpoints", "findings"],
    }
    
    words = sentence.split()
    result_words = words.copy()
    
    # Find replaceable positions
    replaceable = []
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,;:()')
        # Check if word should be preserved
        is_preserved = any(re.search(p, word, re.I) for p in preserve_patterns)
        if not is_preserved and word_lower in synonyms:
            replaceable.append((i, word_lower, word))
    
    # Replace up to n words
    random.shuffle(replaceable)
    for i, word_lower, original in replaceable[:n_replacements]:
        # Preserve punctuation
        prefix = ""
        suffix = ""
        if original and not original[0].isalnum():
            prefix = original[0]
        if original and not original[-1].isalnum():
            suffix = original[-1]
        
        replacement = random.choice(synonyms[word_lower])
        # Match case
        if original[0].isupper():
            replacement = replacement.capitalize()
        
        result_words[i] = prefix + replacement + suffix
    
    return " ".join(result_words)


def augment_sentence_templates(sentence: str, label: str) -> list[str]:
    """Generate variations using templates based on label."""
    augmented = []
    
    if label == "positive":
        # Add emphasis patterns
        patterns = [
            ("is associated with", "has been strongly associated with"),
            ("associated with", "significantly associated with"),
            ("linked to", "strongly linked to"),
            ("correlated with", "positively correlated with"),
        ]
        for old, new in patterns:
            if old in sentence.lower():
                augmented.append(re.sub(re.escape(old), new, sentence, flags=re.I))
                break
    
    elif label == "negative":
        patterns = [
            ("not associated", "showed no association"),
            ("no significant", "failed to show significant"),
            ("did not", "failed to"),
        ]
        for old, new in patterns:
            if old in sentence.lower():
                augmented.append(re.sub(re.escape(old), new, sentence, flags=re.I))
                break
    
    return augmented


def augment_dataset(
    input_path: Path,
    output_path: Path,
    augment_factor: int = 2,
    balance_classes: bool = True,
):
    """Augment training dataset."""
    with open(input_path) as f:
        data = json.load(f)
    
    print(f"Original dataset: {len(data)} samples")
    label_counts = Counter(item["label"] for item in data)
    print(f"Distribution: {dict(label_counts)}")
    
    augmented = list(data)  # Keep originals
    seen = {item["sentence"][:80] for item in data}
    
    # Determine augmentation targets
    if balance_classes:
        max_count = max(label_counts.values())
        target_counts = {label: max_count for label in label_counts}
    else:
        target_counts = {label: count * augment_factor for label, count in label_counts.items()}
    
    # Augment each class
    for label in label_counts:
        class_samples = [item for item in data if item["label"] == label]
        current = label_counts[label]
        target = target_counts[label]
        needed = target - current
        
        print(f"\n{label}: need {needed} more samples")
        
        added = 0
        attempts = 0
        max_attempts = needed * 10
        
        while added < needed and attempts < max_attempts:
            attempts += 1
            sample = random.choice(class_samples)
            
            # Try synonym replacement
            aug_sent = augment_synonym_replacement(sample["sentence"])
            if aug_sent[:80] not in seen and aug_sent != sample["sentence"]:
                augmented.append({"sentence": aug_sent, "label": label, "augmented": True})
                seen.add(aug_sent[:80])
                added += 1
                continue
            
            # Try template augmentation
            template_augs = augment_sentence_templates(sample["sentence"], label)
            for aug_sent in template_augs:
                if aug_sent[:80] not in seen:
                    augmented.append({"sentence": aug_sent, "label": label, "augmented": True})
                    seen.add(aug_sent[:80])
                    added += 1
                    break
        
        print(f"  Added {added} augmented samples")
    
    # Save
    with open(output_path, "w") as f:
        json.dump(augmented, f, indent=2)
    
    print(f"\nAugmented dataset: {len(augmented)} samples")
    final_counts = Counter(item["label"] for item in augmented)
    print(f"Final distribution: {dict(final_counts)}")
    print(f"Saved to {output_path}")
    
    return augmented


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/labeled.json")
    parser.add_argument("--output", default="data/labeled_augmented.json")
    parser.add_argument("--factor", type=int, default=2, help="Augmentation factor")
    parser.add_argument("--no-balance", action="store_true", help="Don't balance classes")
    args = parser.parse_args()
    
    augment_dataset(
        Path(args.input),
        Path(args.output),
        augment_factor=args.factor,
        balance_classes=not args.no_balance,
    )


if __name__ == "__main__":
    main()
