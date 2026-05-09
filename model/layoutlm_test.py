from transformers import LayoutLMv3Processor
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load processor
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)

# Load image
image = Image.open('static/uploads/invoice image.jpeg')

# OCR extraction
ocr_data = pytesseract.image_to_data(
    image,
    output_type=pytesseract.Output.DICT
)

words = []
boxes = []

n = len(ocr_data['text'])

for i in range(n):

    word = ocr_data['text'][i]

    if word.strip() != "":

        words.append(word)

        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]

        boxes.append([x, y, x + w, y + h])

# Process for LayoutLM
encoding = processor(
    image,
    words,
    boxes=boxes,
    return_tensors="pt"
)

print("LayoutLM Processing Successful!")

print(encoding.keys())