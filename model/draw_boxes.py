import pytesseract
from PIL import Image, ImageDraw

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = 'static/uploads/invoice image.jpeg'

image = Image.open(image_path)

draw = ImageDraw.Draw(image)

data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

n = len(data['text'])

for i in range(n):

    word = data['text'][i]

    if word.strip() != "":

        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]

        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline='red',
            width=2
        )

output_path = 'static/uploads/output_boxes.jpg'

image.save(output_path)

print("Bounding boxes image saved successfully!")