# InvoiceAI — AI Invoice Intelligence

InvoiceAI is a Flask application for extracting invoice fields from scanned images and PDFs. It combines Tesseract OCR and OpenCV preprocessing with transparent rule-based field extraction and an optional Hugging Face transformer NER model.

## Features

- Upload PDF, PNG, JPG/JPEG, TIFF, and BMP invoices (up to 20 MB).
- Extract OCR text and rule-based invoice number, date, supplier, subtotal, tax, and total fields.
- Show highlighted OCR regions, NER entities, and missing-field warnings.
- Keep a persistent local invoice history in SQLite.
- Search and filter history, view invoice details, and export individual JSON or Excel results and a combined Excel report.
- Display dashboard and report counts from saved records rather than sample metrics.

## Requirements

- Python 3.9 or newer.
- Tesseract OCR.
- Poppler utilities for PDF conversion.
- Python dependencies listed in `requirements.txt`.

### Install system packages

GitHub Codespaces / Debian / Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

macOS with Homebrew:

```bash
brew install tesseract poppler
```

Windows: install Tesseract OCR and Poppler, then set `TESSERACT_CMD` to the Tesseract executable path and add Poppler's `bin` directory to `PATH`.

## Install and run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. In Codespaces, forward port `5000` when prompted. To use another port, set `PORT` before starting the app:

```bash
PORT=8000 python app.py
```

The NER model (`dslim/bert-base-NER` by default) loads once in the background during application startup so the web server can accept requests while model weights initialize. Transformers may download its model files on first launch; if it cannot load, OCR and rule-based extraction remain available and the results page reports the NER limitation. To select another Hugging Face NER model, set `INVOICE_NER_MODEL`.

OCR preprocessing compares the original grayscale scan and an adaptive-threshold candidate using Tesseract page segmentation modes 6 and 11, then chooses the OCR result with the strongest confidence/word coverage. The highlighted preview is drawn over the original image, and raw recognized text is stored for review. To log full OCR text for a debugging session, set both `LOG_LEVEL=DEBUG` and `LOG_OCR_TEXT=true`; those logs contain sensitive invoice contents and should be handled accordingly.

When the extraction rules change, saved invoice fields are recomputed at application startup from the OCR text already stored in SQLite. The original stored OCR text is not modified; an invoice must be re-uploaded only if improved OCR (rather than improved field mapping) is required.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `5000` | Flask listening port |
| `HOST` | `0.0.0.0` | Listening interface |
| `FLASK_DEBUG` | `false` | Enables Flask debug mode only when set to `true` |
| `FLASK_SECRET_KEY` | Random per process | Set a private stable random value for deployed environments |
| `TESSERACT_CMD` | Tesseract on `PATH` | Explicit Tesseract executable path |
| `INVOICE_NER_MODEL` | `dslim/bert-base-NER` | Hugging Face token-classification model |
| `LOG_LEVEL` | `INFO` | Server log level |

For example, generate a session secret before deployment:

```bash
export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

Do not enable debug mode on a public deployment.

## Data and limitations

Invoice metadata, extracted fields, OCR text, NER output, and warnings are stored in `results/invoices.sqlite3`. The uploaded source file is removed after processing; highlighted preview images are retained under `static/uploads/` for the history detail pages. Remove the corresponding database record and preview file when deleting a document's retained data.

Field extraction uses OCR text and label-aware regular expressions, not a specialized invoice understanding model. Unreadable or absent fields are displayed as missing and flagged for review; no confidence score is invented. NER output and its model score are kept separate from the invoice-field extraction. PDFs are limited to their first page. OCR quality depends on image quality and invoice layout.

## Routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Dashboard and invoice upload |
| `/upload` | POST | Validate and process an invoice |
| `/invoices/<id>` | GET | View saved fields, OCR text, NER, and preview |
| `/history` | GET | Search and filter invoice history |
| `/analytics` | GET | Saved processing metrics and reports |
| `/about` | GET | Processing workflow and local-data notes |
| `/export/<id>.json` | GET | Download one invoice result |
| `/export.xlsx` | GET | Download one invoice (`?id=<id>`) or all saved results |
