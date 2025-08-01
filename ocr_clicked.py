import cv2
import pytesseract
import time


def run_ocr_tesseract(image):
    """
    Run Tesseract OCR on the input image.
    :param image: BGR image from OpenCV
    :return: recognized text
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    # Apply binary thresholding (optional but helps OCR)
    

    # Start timing
    start_time = time.time()

    # OCR configuration
    custom_config = r'--oem 1 --psm 3'

    # Perform OCR
    text = pytesseract.image_to_string(thresh, config=custom_config)

    # Time taken
    elapsed = time.time() - start_time

    # Show result
    print("\n  OCR Result:")
    print(text.strip())
    print(f" Inference time: {elapsed:.2f} sec\n")

    return text

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Webcam started. Press 'c' to run OCR, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to grab frame")
            break

        # Optional: show live video
        cv2.imshow("Press 'c' to OCR, 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("  Running OCR on current frame...")
            run_ocr_tesseract(frame)

        elif key == ord('q'):
            print("  Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

