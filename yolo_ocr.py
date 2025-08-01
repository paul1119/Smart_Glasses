
import cv2
import pytesseract
import time
import sys
import select
import tty
import termios
from ultralytics import YOLO

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

    for box in results.boxes:
        found = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        print(f"\n📚 Detected 'book' at {x1},{y1},{x2},{y2}")

        # start = time.time()
        text = run_ocr(crop)
        # duration = time.time() - start
        # cv2.imshow("testing", frame)
        
        print("🧠 OCR Result:")
        print(text if text else "(No text detected)")
        # print(f"⏱️ OCR took {duration:.2f} sec")
        print("=" * 40)
        #time.sleep(1)
    if not found:
        print("📭 No 'book' detected.")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("📷 Webcam started. Press 'c' to detect 'book' and run OCR, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        detect_and_ocr(frame)
        cv2.imshow("testing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        """
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            print("🔍 Running detection + OCR...")
            detect_and_ocr(frame)
        elif key == ord('q'):
            print("👋 Exiting...")
            break
          
        # Show the frame
        cv2.imshow("Live Feed (press 'c' for OCR, 'q' to quit)", frame) 
        """

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
