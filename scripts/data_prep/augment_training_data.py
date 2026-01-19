"""Augment training data with negative and no_association samples from corpus."""
import csv
import json
import re
from pathlib import Path
from collections import Counter


def extract_negative_sentences(corpus_path: Path) -> list[dict]:
    """Extract sentences with negative findings from abstracts."""
    keywords = [
        'not associated with',
        'no significant difference',
        'no significant association',
        'failed to demonstrate',
        'failed to show',
        'did not improve',
        'did not reduce',
        'did not affect',
        'no benefit',
        'no effect on',
        'not a predictor',
        'was not significant',
        'were not significant',
        'no prognostic benefit',
        'no mortality benefit',
        'unlikely to be',
    ]
    
    found = []
    seen = set()
    
    with open(corpus_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            abstract = row.get('abstract', '').strip()
            if not abstract:
                continue
            abstract_lower = abstract.lower()
            if 'hfpef' not in abstract_lower and 'preserved ejection fraction' not in abstract_lower:
                continue
                
            for kw in keywords:
                if kw in abstract_lower:
                    # Find the sentence containing the keyword
                    idx = abstract_lower.find(kw)
                    # Find sentence boundaries
                    start = abstract.rfind('.', 0, idx)
                    start = start + 1 if start != -1 else 0
                    end = abstract.find('.', idx)
                    end = end + 1 if end != -1 else len(abstract)
                    
                    sentence = abstract[start:end].strip()
                    
                    # Quality filters
                    if len(sentence) < 40 or len(sentence) > 500:
                        continue
                    if sentence[:50] in seen:
                        continue
                    # Must mention HFpEF in or near the sentence
                    context = abstract_lower[max(0,start-100):min(len(abstract),end+100)]
                    if 'hfpef' not in context and 'preserved ejection' not in context:
                        continue
                        
                    seen.add(sentence[:50])
                    found.append({
                        'sentence': sentence,
                        'label': 'negative',
                        'source': 'abstract',
                        'keyword': kw,
                    })
                    break
    
    return found


def extract_no_association_sentences(corpus_path: Path) -> list[dict]:
    """Extract method/background sentences that mention HFpEF without claims."""
    patterns = [
        (r'^This (retrospective|prospective|observational|cross-sectional)', 'study_design'),
        (r'^(We|The authors) (aimed|sought|investigated|examined|evaluated)', 'objective'),
        (r'^(METHODS|OBJECTIVE|AIM|BACKGROUND|PURPOSE):', 'section_header'),
        (r'patients (were|who were) (included|enrolled|recruited|randomized)', 'enrollment'),
        (r'Search (terms|strategy)', 'search_methods'),
        (r'data (were|was) (collected|extracted|obtained)', 'data_collection'),
    ]
    
    found = []
    seen = set()
    
    with open(corpus_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check evidence_sentence first
            sentence = row.get('evidence_sentence', '').strip()
            if sentence and sentence[:50] not in seen:
                sent_lower = sentence.lower()
                for pat, pat_type in patterns:
                    if re.search(pat, sentence, re.IGNORECASE):
                        if 'hfpef' in sent_lower or 'preserved ejection' in sent_lower:
                            seen.add(sentence[:50])
                            found.append({
                                'sentence': sentence,
                                'label': 'no_association', 
                                'source': 'evidence',
                                'pattern_type': pat_type,
                            })
                            break
            
            # Also check abstract for method sentences
            abstract = row.get('abstract', '').strip()
            if not abstract:
                continue
            
            # Split into sentences and check first few
            sentences = re.split(r'(?<=[.!?])\s+', abstract)
            for sent in sentences[:3]:
                if sent[:50] in seen or len(sent) < 30:
                    continue
                sent_lower = sent.lower()
                if 'hfpef' not in sent_lower and 'preserved ejection' not in sent_lower:
                    continue
                for pat, pat_type in patterns:
                    if re.search(pat, sent, re.IGNORECASE):
                        seen.add(sent[:50])
                        found.append({
                            'sentence': sent,
                            'label': 'no_association',
                            'source': 'abstract',
                            'pattern_type': pat_type,
                        })
                        break
    
    return found


def main():
    corpus_path = Path('data/hfpef_corpus.csv')
    labeled_path = Path('data/labeled.json')
    output_path = Path('data/labeled_augmented.json')
    
    # Load existing labeled data
    with open(labeled_path) as f:
        existing = json.load(f)
    
    existing_sents = {item['sentence'][:50] for item in existing}
    label_counts = Counter(item['label'] for item in existing)
    print(f"Existing data: {dict(label_counts)}")
    
    # Extract new samples
    negative_samples = extract_negative_sentences(corpus_path)
    no_assoc_samples = extract_no_association_sentences(corpus_path)
    
    print(f"Found {len(negative_samples)} negative candidates")
    print(f"Found {len(no_assoc_samples)} no_association candidates")
    
    # Add new samples (avoiding duplicates)
    added_neg = 0
    added_no_assoc = 0
    
    for sample in negative_samples:
        if sample['sentence'][:50] not in existing_sents:
            existing.append({
                'sentence': sample['sentence'],
                'label': 'negative',
            })
            existing_sents.add(sample['sentence'][:50])
            added_neg += 1
            if added_neg >= 50:  # Limit to balance classes
                break
    
    for sample in no_assoc_samples:
        if sample['sentence'][:50] not in existing_sents:
            existing.append({
                'sentence': sample['sentence'],
                'label': 'no_association',
            })
            existing_sents.add(sample['sentence'][:50])
            added_no_assoc += 1
            if added_no_assoc >= 50:  # Limit to balance classes
                break
    
    print(f"Added {added_neg} negative samples")
    print(f"Added {added_no_assoc} no_association samples")
    
    # Final counts
    final_counts = Counter(item['label'] for item in existing)
    print(f"\nFinal distribution: {dict(final_counts)}")
    print(f"Total samples: {len(existing)}")
    
    # Save augmented data
    with open(output_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    # Also update the main labeled.json
    with open(labeled_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Updated {labeled_path}")


if __name__ == '__main__':
    main()
