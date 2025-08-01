import cv2
# import easyocr
# import keras_ocr
import pytesseract

# --- Load and preprocess image ---
image_path = 'camera_test.png'
img = cv2.imread(image_path)

# pipeline = keras_ocr.pipeline.Pipeline()

#predictions = pipeline.recognize(image='camera_test.png')

# reader = easyocr.Reader(['en'])

# result = reader.readtext('camera_test.png', detail=0)
config = '--oem 1 --psm 3'
text = pytesseract.image_to_string(img, config=config)

# --- Output result ---
print("🔍 OCR Result:\n")
print(text)
