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
    top_associated_words: list
    top_not_associated_words: list
    explanation: str


def explain_prediction(sentence: str, classifier) -> ExplanationResult:
    """
    Generate human-readable explanation for a prediction.
    
    Uses keyword matching to identify likely triggers for classification.
    """
    # Get basic prediction
    label, confidence = classifier.predict(sentence)
    
    # Define trigger patterns for each class
    ASSOCIATED_TRIGGERS = [
        'elevated', 'increased', 'higher', 'associated', 'correlated',
        'predictor', 'marker', 'linked', 'contributes', 'role',
        'significant', 'risk factor', 'promotes', 'induces'
    ]
    
    NOT_ASSOCIATED_TRIGGERS = [
        'no significant', 'not associated', 'failed', 'did not',
        'no correlation', 'no relationship', 'not significantly',
        'no difference', 'unclear', 'unknown', 'no evidence'
    ]
    
    INCIDENTAL_TRIGGERS = [
        'we aimed', 'we sought', 'we investigated', 'methods',
        'enrolled', 'recruited', 'patients were', 'retrospective',
        'prospective', 'measured', 'analyzed', 'between january'
    ]
    
    sent_lower = sentence.lower()
    
    # Find matching triggers
    found_associated = [t for t in ASSOCIATED_TRIGGERS if t in sent_lower]
    found_not_associated = [t for t in NOT_ASSOCIATED_TRIGGERS if t in sent_lower]
    found_incidental = [t for t in INCIDENTAL_TRIGGERS if t in sent_lower]
    
    # Build explanation based on prediction
    if label == 'associated':
        triggers = found_associated[:3] if found_associated else ['context-based']
        explanation = f"Predicted '{label}' ({confidence:.1%}). "
        explanation += f"Triggers: {', '.join(triggers)}"
        top_associated = [(t, '1.0') for t in triggers]
        top_not_associated = [(t, '0.5') for t in found_not_associated[:2]]
    
    elif label == 'not_associated':
        triggers = found_not_associated[:3] if found_not_associated else ['semantic negation']
        explanation = f"Predicted '{label}' ({confidence:.1%}). "
        explanation += f"Triggers: {', '.join(triggers)}"
        top_associated = [(t, '1.0') for t in triggers]
        top_not_associated = [(t, '0.5') for t in found_associated[:2]]
    
    else:  # incidental
        triggers = found_incidental[:3] if found_incidental else ['methodology language']
        explanation = f"Predicted '{label}' ({confidence:.1%}). "
        explanation += f"Triggers: {', '.join(triggers)}"
        top_associated = [(t, '1.0') for t in triggers]
        top_not_associated = []
    
    return ExplanationResult(
        sentence=sentence,
        prediction=label,
        confidence=confidence,
        top_associated_words=top_associated,
        top_not_associated_words=top_not_associated,
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
            key_words = ', '.join(w for w, _ in r.top_associated_words)
            writer.writerow([
                r.sentence[:200],
                r.prediction,
                f"{r.confidence:.2f}",
                key_words,
                r.explanation
            ])
    
    print(f"Saved explanations to {output_path}")
