import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = Image.open('static/uploads/invoice image.jpeg')

data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

n = len(data['text'])

for i in range(n):

    word = data['text'][i]

    if word.strip() != "":

        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]

        print(f"Word: {word}")
        print(f"Box: ({x}, {y}, {w}, {h})")
        print("-------------------")