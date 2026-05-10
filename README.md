# Scan2Struct - AI Invoice Intelligence

An advanced web-based invoice processing system using Visual Named Entity Recognition (NER) to automatically extract key information from scanned invoices.

## Features

- **📄 Invoice Upload**: Support for both image and PDF formats
- **🎯 Smart Extraction**: Automatically extracts Invoice Number, Date, and Total Amount
- **🤖 AI Entity Recognition**: Uses transformer-based NER models to identify and classify all entities
- **📊 OCR Processing**: Advanced OCR with bounding box visualization
- **💾 Data Export**: Export results to JSON and Excel formats
- **📈 Analytics**: Track processing metrics and performance
- **📜 History**: View all processed invoices

## Project Structure

```
scanned-invoices-visual-ner/
├── app.py                 # Flask application entry point
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates
│   ├── index.html        # Upload page
│   ├── entities.html     # Results page
│   ├── analytics.html    # Analytics dashboard
│   └── history.html      # Processing history
├── static/               # Static files
│   └── uploads/          # Temporary uploaded files
├── results/              # Output directory
│   ├── output.json       # JSON export
│   └── output.xlsx       # Excel export
├── model/                # ML model utilities
│   ├── draw_boxes.py
│   ├── labels.py
│   ├── layoutlm_test.py
│   ├── ocr_boxes.py
│   └── predict_entities.py
└── dataset/              # Training data directory
```

## Installation

### Prerequisites
- Python 3.8+
- Tesseract OCR engine
- poppler (for PDF processing)

### Setup

1. Clone or download the project:
```bash
cd scanned-invoices-visual-ner
```

2. Create a virtual environment:
```bash
python -m venv venv_new
```

3. Activate the virtual environment:

**Windows (PowerShell):**
```powershell
.\venv_new\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv_new\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv_new/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Configure OCR paths in `app.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

3. Use the interface to:
   - **Upload Invoice**: Upload an image or PDF invoice
   - **View Results**: See extracted data and entity predictions
   - **Check Analytics**: View processing metrics
   - **Browse History**: Review previous processed invoices

## Key Technologies

- **Flask**: Web framework
- **PyTorch & Hugging Face Transformers**: NER models
- **OpenCV**: Image processing
- **Tesseract OCR**: Optical character recognition
- **Pandas**: Data manipulation
- **pdf2image**: PDF handling

## API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home page - Invoice upload |
| `/upload` | POST | Process uploaded invoice |
| `/analytics` | GET | Analytics dashboard |
| `/history` | GET | Processing history |

## Output Files

After processing, the following files are generated:

- **output.json**: Invoice data and predictions in JSON format
- **output.xlsx**: Invoice data in Excel format
- **highlighted_invoice.png**: Invoice with detected text boxes highlighted

## Confidence Scores

The system provides confidence scores for:
- Invoice Number extraction (85-99%)
- Invoice Date extraction (85-97%)
- Total Amount extraction (85-99%)

## Known Limitations

- PDF processing requires poppler installation
- Large images may take longer to process
- OCR accuracy depends on image quality
- NER model performance varies with invoice format

## Future Enhancements

- [ ] Batch invoice processing
- [ ] Custom model training
- [ ] Multi-language support
- [ ] Database integration
- [ ] Real-time monitoring dashboard
- [ ] Email integration

## Troubleshooting

### Missing Tesseract Error
Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki

### PDF Processing Error
Install poppler: `https://poppler.freedesktop.org/`

### Module Import Errors
Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

## License

This project is provided as-is for invoice processing and automation.

## Support

For issues or questions, please refer to the documentation or check the console output for detailed error messages.
