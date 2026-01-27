# Benchmark Datasets

## Evaluated

### BioRED (completed)
- **Source**: https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/
- **Size**: 600 PubMed abstracts, 274 gene-disease relations in test set
- **Result**: 94% accuracy on association detection
- **Caveat**: Task mismatch - BioRED distinguishes causal vs protective, we detect existence of association
- **Files**: `biored/` directory

## Recommended (public gene-disease)

### EU-ADR Corpus (gene-disease subset)
- **Paper**: https://pmc.ncbi.nlm.nih.gov/articles/PMC4558020/
- **Size**: 100 MEDLINE abstracts for gene-disease relations
- **Labels**: Positive Association (PA), Negative Association (NA), Speculative Association (SA), False Association (FA)
- **Why better match**: Includes explicit negative and false association labels
- **Mapping**:
  - PA -> Associated
  - NA/FA -> Not_Associated
  - SA -> Incidental

## Recommended (requires DUA)

### i2b2/n2c2 2010 Assertion Dataset
- **Source**: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Paper**: https://pmc.ncbi.nlm.nih.gov/articles/PMC3168320/
- **Size**: 394 training + 477 test clinical notes
- **Why better match**: Has "absent" category = our "Not_Associated"

Assertion categories:
| i2b2 Label | Our Label | Meaning |
|------------|-----------|---------|
| present | Associated | Condition confirmed |
| absent | Not_Associated | Condition ruled out |
| hypothetical | Incidental | Hypothetical mention |
| possible | Incidental | Uncertain mention |
| conditional | Incidental | Conditional mention |

**To access**: Submit DUA at https://n2c2.dbmi.hms.harvard.edu/data-sets

## Label Mapping Reference

Old -> New terminology:
- Positive -> Associated
- Negative -> Not_Associated
- Neutral -> Incidental
