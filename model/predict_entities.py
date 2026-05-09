import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = Image.open('static/uploads/invoice image.jpeg')

data = pytesseract.image_to_data(
    image,
    output_type=pytesseract.Output.DICT
)

words = []

n = len(data['text'])

for i in range(n):

    word = data['text'][i]

    if word.strip() != "":

        words.append(word)

# Simulated Entity Prediction
predictions = []

for word in words:

    label = "O"

    if "invoice" in word.lower():
        label = "B-INVOICE_NO"

    elif "$" in word:
        label = "B-TOTAL"

    elif word.lower() in ["oct", "jan", "feb", "mar"]:
        label = "B-DATE"

    predictions.append((word, label))

# Display Results
for word, label in predictions:

    print(f"{word:<20} --> {label}")