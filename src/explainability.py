"""
Model explainability via keyword-based trigger detection.

Identifies which words likely triggered the classification decision.
"""
from dataclasses import dataclass


@dataclass
class ExplanationResult:
    sentence: str
    prediction: str
    confidence: float
    top_positive_words: list
    top_negative_words: list
    explanation: str


def explain_prediction(sentence: str, classifier) -> ExplanationResult:
    """
    Generate human-readable explanation for a prediction.
    
    Uses keyword matching to identify likely triggers for classification.
    """
    # Get basic prediction
    label, confidence = classifier.predict(sentence)
    
    # Define trigger patterns for each class
    POSITIVE_TRIGGERS = [
        'elevated', 'increased', 'higher', 'associated', 'correlated',
        'predictor', 'marker', 'linked', 'contributes', 'role',
        'significant', 'risk factor', 'promotes', 'induces'
    ]
    
    NEGATIVE_TRIGGERS = [
        'no significant', 'not associated', 'failed', 'did not',
        'no correlation', 'no relationship', 'not significantly',
        'no difference', 'unclear', 'unknown', 'no evidence'
    ]
    
    NO_ASSOC_TRIGGERS = [
        'we aimed', 'we sought', 'we investigated', 'methods',
        'enrolled', 'recruited', 'patients were', 'retrospective',
        'prospective', 'measured', 'analyzed', 'between january'
    ]
    
    sent_lower = sentence.lower()
    
    # Find matching triggers
    found_positive = [t for t in POSITIVE_TRIGGERS if t in sent_lower]
    found_negative = [t for t in NEGATIVE_TRIGGERS if t in sent_lower]
    found_no_assoc = [t for t in NO_ASSOC_TRIGGERS if t in sent_lower]
    
    # Build explanation based on prediction
    if label == 'positive':
        triggers = found_positive[:3] if found_positive else ['context-based']
        explanation = f"Predicted '{label}' ({confidence:.1%}). "
        explanation += f"Triggers: {', '.join(triggers)}"
        top_positive = [(t, '1.0') for t in triggers]
        top_negative = [(t, '0.5') for t in found_negative[:2]]
    
    elif label == 'negative':
        triggers = found_negative[:3] if found_negative else ['semantic negation']
        explanation = f"Predicted '{label}' ({confidence:.1%}). "
        explanation += f"Triggers: {', '.join(triggers)}"
        top_positive = [(t, '1.0') for t in triggers]
        top_negative = [(t, '0.5') for t in found_positive[:2]]
    
    else:  # no_association
        triggers = found_no_assoc[:3] if found_no_assoc else ['methodology language']
        explanation = f"Predicted '{label}' ({confidence:.1%}). "
        explanation += f"Triggers: {', '.join(triggers)}"
        top_positive = [(t, '1.0') for t in triggers]
        top_negative = []
    
    return ExplanationResult(
        sentence=sentence,
        prediction=label,
        confidence=confidence,
        top_positive_words=top_positive,
        top_negative_words=top_negative,
        explanation=explanation
    )


def explain_batch(sentences: list, classifier) -> list:
    """Explain multiple predictions."""
    return [explain_prediction(s, classifier) for s in sentences]


def generate_explanation_report(sentences: list, classifier, output_path: str):
    """Generate CSV report with explanations for each sentence."""
    import csv
    
    results = explain_batch(sentences, classifier)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence', 'prediction', 'confidence', 'key_words', 'explanation'])
        
        for r in results:
            key_words = ', '.join(w for w, _ in r.top_positive_words)
            writer.writerow([
                r.sentence[:200],
                r.prediction,
                f"{r.confidence:.2f}",
                key_words,
                r.explanation
            ])
    
    print(f"Saved explanations to {output_path}")
