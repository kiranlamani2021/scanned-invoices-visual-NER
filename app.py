from flask import Flask, render_template, request
import os
import pytesseract
from PIL import Image
import re
import json
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():

    if 'invoice' not in request.files:
        return "No file uploaded"

    file = request.files['invoice']

    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(filepath)

    # OCR Extraction
    image = Image.open(filepath)

    extracted_text = pytesseract.image_to_string(image)

    # Invoice Details Extraction
    invoice_number = re.search(
        r'Invoice[#:]?\s*[:#]?\s*(\d+)',
        extracted_text,
        re.IGNORECASE
    )

    invoice_date = re.search(
        r'Invoice date[:\s]*(\w+\s\d{1,2},\s\d{4})',
        extracted_text,
        re.IGNORECASE
    )

    total_amount = re.search(
        r'SUBTOTAL\s*\$?(\d+\.\d+)',
        extracted_text,
        re.IGNORECASE
    )

    invoice_number = invoice_number.group(1) if invoice_number else "Not Found"

    invoice_date = invoice_date.group(1) if invoice_date else "Not Found"

    total_amount = total_amount.group(1) if total_amount else "Not Found"

    # Simulated NER Prediction
    words = extracted_text.split()

    predictions = []

    for word in words:

        label = "O"

        if word.isdigit():

            label = "B-INVOICE_NO"

        elif "invoice" in word.lower():

            label = "B-INVOICE_NO"

        elif "$" in word:

            label = "B-TOTAL"

        elif word.lower() in ["oct", "jan", "feb", "mar"]:

            label = "B-DATE"

        predictions.append((word, label))

    # JSON Export
    result_data = {
        "Invoice_Number": invoice_number,
        "Invoice_Date": invoice_date,
        "Total_Amount": total_amount,
        "Predictions": predictions
    }

    with open('results/output.json', 'w') as json_file:

        json.dump(result_data, json_file, indent=4)

    # Excel Export
    excel_data = {
        "Invoice Number": [invoice_number],
        "Invoice Date": [invoice_date],
        "Total Amount": [total_amount]
    }

    df = pd.DataFrame(excel_data)

    df.to_excel('results/output.xlsx', index=False)

    return render_template(
    'entities.html',
    predictions=predictions,
    invoice_number=invoice_number,
    invoice_date=invoice_date,
    total_amount=total_amount,
    extracted_text=extracted_text,
    image_file=file.filename
)


if __name__ == '__main__':
    app.run(debug=True)