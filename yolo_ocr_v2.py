import cv2
import pytesseract
import time
import sys
import select
import tty
import termios
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO("best_text_area.pt")

def run_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    config = "--oem 1 --psm 3"
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

def get_key(timeout=0.1):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ''

def detect_and_ocr(frame):
    results = model(frame)[0]
    found = False

    # Ensure it's a proper 2D array
    boxes = np.array([box.xyxy[0].cpu().numpy() for box in results.boxes])

    # Now you can safely index with [:, 0], [:, 1], etc.
    if len(boxes) > 0:
        found = True
        x_min = int(np.min(boxes[:, 0]))
        y_min = int(np.min(boxes[:, 1]))
        x_max = int(np.max(boxes[:, 2]))
        y_max = int(np.max(boxes[:, 3]))

    if not found:
        return

    # Draw the big merged box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imwrite('cropped_image.jpg', frame)
    # Crop merged box
    merged_crop = frame[y_min:y_max, x_min:x_max]

    text = run_ocr(merged_crop)
    print("OCR Result:")
    print(text if text else "(No text detected)")

    print("=" * 40)

    if not found:
        print("No textk' detected.")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Webcam started. Press 'c' to detect 'book' and run OCR, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        #detect_and_ocr(frame)
        #cv2.imshow("testing", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            print(" Running detection + OCR...")
            detect_and_ocr(frame)
        elif key == ord('q'):
            print(" Exiting...")
            break

        # Show the frame
        cv2.imshow("Live Feed (press 'c' for OCR, 'q' to quit)", frame) 


        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
