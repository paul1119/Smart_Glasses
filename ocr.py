import pytesseract

def ocr_function(frame_bgr):
    # Convert to RGB for Tesseract
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)
   print("[OCR] Text:", text.strip())
