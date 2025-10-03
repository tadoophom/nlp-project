Clinical Keyword Polarity Suite
================================

This repository contains a Streamlit-based web application and a companion
FastAPI service for analysing clinical text for mentions of specific keywords
and their polarity (positive vs. negative) using spaCy and optional
medspaCy context rules. The goal of this project is to provide a
production‑ready foundation for medical NLP workflows, including batch
processing, interactive dashboards, API endpoints and extensibility for
active learning and audit logging.

Features
--------

* **Streamlit UI** – drag‑and‑drop TXT/PDF/DOCX/PNG/JPG/ZIP ingestion with OCR
  fallback, multilingual model selection, dark‑mode toggle and interactive
  dependency tree visualisation with hover tooltips.
* **Analytics dashboard** – real‑time session metrics, interactive Plotly
  charts for sentiment distribution and aggregate classification counts across
  documents.
* **FastAPI service** – headless API exposing a `/analyze` endpoint for JSON
  input/output, suitable for integration into microservices or batch jobs.
* **Active learning hooks** – every keyword hit can be annotated with a
  thumbs‑up/down in the UI; feedback is recorded to an SQLite database for
  downstream model refinement (see `database.py`).
* **Packaging and CI** – Dockerfile, docker‑compose and GitHub Actions
  workflow included for reproducible builds and continuous integration. A
  pre‑commit configuration ensures code is type‑checked and linted.

Quick start
-----------

1. **Install dependencies**: this project uses `pyproject.toml` to define its
   dependencies. You can install them via [pipx](https://pypi.org/project/pipx/) or
   `pip` with the following command:

   ```bash
   pip install -e .
   ```

2. **Run the Streamlit UI**:

   ```bash
   streamlit run app.py
   ```

   Open the provided local URL in your browser. Select a spaCy model, choose
   whether to use GPU acceleration, paste or upload a document and specify
   comma‑separated keywords. Click **Analyze** to see results and charts.

3. **Run the FastAPI server**:

   ```bash
   uvicorn api:app --reload
   ```

   Once running, the OpenAPI schema will be available at `/docs`. You can send
   a JSON payload like `{"text": "...", "keywords": ["procalcitonin"]}`
   to `POST /analyze` and receive structured results.

PubMed Abstract Retrieval & Dataset Creation
-------------------------------------------

You can use the PubMed API to search for abstrac  ts using clinical keywords and/or MeSH terms. Retrieved abstracts are saved as a CSV file for downstream analysis and model training.

**Usage:**

1. Run the PubMed fetch script:

  ```bash
  python pubmed_fetch.py --keywords "procalcitonin,sepsis" --mesh "Sepsis"
  ```

  This will create `pubmed_dataset.csv` containing abstracts, titles, and metadata.

2. Use this CSV as input for your NLP pipeline or manual annotation.



Disease–Protein Corpus Builder
------------------------------

For workflows that require sentiment-aware relationships between a disease and
candidate biomarkers (e.g. proteins), use `corpus_pipeline.py`. It combines
PubMed querying (keywords + MeSH) with the existing polarity classifier so that
each abstract is labelled as a positive, negative or neutral co-mention.

**Example (HFpEF vs. protein panel):**

```bash
python3 corpus_pipeline.py \
  --protein-file sample_aktan.xlsx \
  --identifier-column protein \
  --score-column HFpEF \
  --top-n 50 \
  --disease-keyword HFpEF \
  --disease-keyword "Heart Failure with Preserved Ejection Fraction" \
  --extra-mesh "Heart Failure, Diastolic" \
  --logic disease_and_protein \
  --retmax 75 \
  --output data/hfpef_corpus.csv
```

Key options:

* `--disease-keyword` / `--disease-mesh` – repeatable switches to define the
  disease concept across PubMed fields.
* `--extra-keyword` / `--extra-mesh` – optional filters applied to every query
  (e.g. study design terms or comorbidities).
* `--logic` – choose how disease/protein/extra groups are combined. Supported
  values: `disease_and_protein`, `all_and`, `all_or`, `disease_or_protein`.
* `--protein-file` – accepts CSV or XLSX. When a score column exists you can
  restrict with `--top-n` and/or `--min-score`.

The resulting CSV contains one row per (protein, article) pair with PubMed
metadata, the exact query, and polarity evidence so you can audit candidate
relationships quickly or feed them into downstream analytics.

Confusion Matrix for Manual Validation
-------------------------------------

After running classification, you can manually review predictions and generate a confusion matrix to assess accuracy.

**Usage:**

1. Annotate predictions in the CSV (e.g., add a `true_label` column).
2. Run the confusion matrix script:

  ```bash
  python confusion_matrix.py --input pubmed_dataset.csv --pred_col predicted_label --true_col true_label
  ```

  This will display a confusion matrix showing correct/incorrect classifications.

Development
-----------

* The NLP logic (loading models, extracting keyword hits and rendering
  dependency trees) lives in `nlp_utils.py`. This ensures both Streamlit and
  FastAPI reuse the same code paths.
* Test cases reside in the `tests/` directory and can be executed with
  `pytest`.
* The CI pipeline defined in `.github/workflows/ci.yml` installs dependencies,
  runs linters and executes the test suite on every push.

Contributing
------------

Bug reports and feature requests are welcome. When contributing code please
run `pre-commit run --all-files` locally to ensure your changes pass linting
and type checks.