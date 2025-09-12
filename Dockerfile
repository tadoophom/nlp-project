FROM python:3.11-slim

LABEL maintainer="Clinical NLP Team"

# Install system packages required for OCR and PDF support.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir .

EXPOSE 8501 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]