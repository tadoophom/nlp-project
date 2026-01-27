# Annotation Guidelines for Protein-Disease Association Classification

## Task
Classify each sentence based on whether it describes a meaningful association between a protein/gene and a disease (specifically HFpEF or heart failure).

## Labels

### Associated
The sentence provides evidence that the protein IS related to the disease.

**Criteria:**
- Protein is described as a biomarker, risk factor, or therapeutic target
- Protein expression/levels are linked to disease state
- Protein function is connected to disease mechanism
- Causal OR protective relationships both count as "Associated"

**Examples:**
- "Elevated BNP levels are associated with worse outcomes in HFpEF" → **Associated**
- "ACE inhibitors reduce mortality in heart failure patients" → **Associated**
- "IL-6 protects against cardiac dysfunction" → **Associated** (protective = still associated)

### Not_Associated
The sentence explicitly states there is NO relationship.

**Criteria:**
- Study found no significant association
- Relationship was ruled out or refuted
- Negative study results ("failed to demonstrate", "no effect")

**Examples:**
- "There was no significant association between adiponectin and HFpEF" → **Not_Associated**
- "MRA use was not associated with improved outcomes" → **Not_Associated**
- "The study failed to demonstrate a benefit" → **Not_Associated**

### Incidental
The sentence mentions both protein and disease but makes no claim about their relationship.

**Criteria:**
- Co-occurrence without causal claim
- Study population description only
- Methods section mentioning protein as measurement
- Hypothetical/conditional statements without evidence

**Examples:**
- "This study included 400 HFpEF patients who underwent PCI" → **Incidental**
- "We measured troponin levels in the cohort" → **Incidental**
- "If cardiac biomarkers increase, further testing may be needed" → **Incidental**

## Special Cases

### Mixed signals
If a sentence contains BOTH positive and negative signals (e.g., "failed to show X but suggested Y"), label based on the PRIMARY conclusion of the sentence.

### Uncertain language
- "may be associated" → **Incidental** (hypothesis, not evidence)
- "was associated" → **Associated** (evidence)
- "was not associated" → **Not_Associated** (negative evidence)

## Confidence Rating
Rate your confidence for each annotation:
- **3** = Clear, unambiguous
- **2** = Reasonable interpretation, minor ambiguity
- **1** = Difficult, could go either way

## Notes Field
Use for:
- Explaining difficult decisions
- Flagging sentences that need discussion
- Noting any issues with the text
