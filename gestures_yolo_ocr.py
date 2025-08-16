import cv2
import time
import math
import threading
import mediapipe as mp
from ultralytics import YOLO

# ===================== YOLO (Thumb Up trigger) =====================
YOLO_WEIGHTS = "yolov8n.pt"
yolo_model = YOLO(YOLO_WEIGHTS)

# ===================== Your OCR function ===========================
# If you already have one, just import it, e.g.:
# from my_ocr_module import ocr_function
#
# Fallback demo implementation (uses pytesseract if available):
try:
    import pytesseract
    def ocr_function(frame_bgr):
        # Convert to RGB for Tesseract
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(rgb)
        print("[OCR] Text:", text.strip())
except Exception:
    def ocr_function(frame_bgr):
        # Stub if pytesseract not installed or you have your own function
        print("[OCR] (stub) OCR function called")

# ===================== MediaPipe setup =============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ===================== Helpers ====================================
def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def bbox_size(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(max(xs)-min(xs), max(ys)-min(ys))

# ---- OK gesture ----
def is_ok_gesture(landmarks, w, h):
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP, MIDDLE_PIP = 12, 10
    RING_TIP,   RING_PIP   = 16, 14
    PINKY_TIP,  PINKY_PIP  = 20, 18

    hand_scale = max(20.0, bbox_size(pts))
    thumb_index_dist = l2(pts[THUMB_TIP], pts[INDEX_TIP])
    close_enough = (thumb_index_dist / hand_scale) < 0.18

    middle_extended = pts[MIDDLE_TIP][1] < pts[MIDDLE_PIP][1]
    ring_extended   = pts[RING_TIP][1]   < pts[RING_PIP][1]
    pinky_extended  = pts[PINKY_TIP][1]  < pts[PINKY_PIP][1]
    return close_enough and middle_extended and ring_extended and pinky_extended

# ---- Thumbs up gesture ----
def is_thumb_up_gesture(landmarks, w, h):
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]
    WRIST = 0
    THUMB_MCP, THUMB_TIP = 2, 4
    INDEX_PIP, INDEX_TIP = 6, 8
    MIDDLE_PIP, MIDDLE_TIP = 10, 12
    RING_PIP, RING_TIP = 14, 16
    PINKY_PIP, PINKY_TIP = 18, 20

    hand_scale = max(20.0, bbox_size(pts))
    v = (pts[THUMB_TIP][0] - pts[THUMB_MCP][0], pts[THUMB_TIP][1] - pts[THUMB_MCP][1])
    v_len = max(1e-6, math.hypot(v[0], v[1]))
    cos_to_up = -(v[1]) / v_len
    thumb_long_enough = (v_len / hand_scale) > 0.25
    thumb_up = (cos_to_up > 0.6) and thumb_long_enough

    wr = pts[WRIST]
    def folded(tip, pip): return l2(pts[tip], wr) < 0.9 * l2(pts[pip], wr)
    others_folded = all([
        folded(INDEX_TIP, INDEX_PIP),
        folded(MIDDLE_TIP, MIDDLE_PIP),
        folded(RING_TIP,   RING_PIP),
        folded(PINKY_TIP,  PINKY_PIP),
    ])

    not_ok_like = (l2(pts[THUMB_TIP], pts[INDEX_TIP]) / hand_scale) > 0.22
    return thumb_up and others_folded and not_ok_like

# ===================== Actions (async) =============================
# -- YOLO detection --
_det_lock = threading.Lock()
_det_active = False
def object_detection(frame, show_window=True):
    results = yolo_model.predict(source=frame, imgsz=416, conf=0.25, device="cpu", verbose=False)
    r = results[0]
    for b in r.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        print(f"[YOLO] {r.names[cls_id]} {conf:.2f} @ ({x1},{y1},{x2},{y2})")
    if show_window:
        annotated = r.plot()
        cv2.imshow("YOLOv8 detections", annotated)
        cv2.waitKey(1)

def start_detection_async(frame, show_window=True):
    global _det_active
    with _det_lock:
        if _det_active: return
        _det_active = True
    def worker(img):
        global _det_active
        try:
            object_detection(img, show_window)
        finally:
            with _det_lock:
                _det_active = False
    threading.Thread(target=worker, args=(frame.copy(),), daemon=True).start()

# -- OCR --
_ocr_lock = threading.Lock()
_ocr_active = False
def start_ocr_async(frame, show_window=False):
    """Runs your ocr_function(frame) in background once per trigger."""
    global _ocr_active
    with _ocr_lock:
        if _ocr_active: return
        _ocr_active = True
    def worker(img):
        global _ocr_active
        try:
            ocr_function(img)  # <-- call your OCR here
        finally:
            with _ocr_lock:
                _ocr_active = False
    threading.Thread(target=worker, args=(frame.copy(),), daemon=True).start()

# ===================== Main loop ==================================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    show_window = False
    cooldown_print_s = 0.8
    cooldown_yolo_trigger_s = 2.0
    cooldown_ocr_trigger_s  = 2.0

    last_print_ok = 0.0
    last_thumb_trigger = 0.0
    last_ok_trigger = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to grab frame.")
                break

            # frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)

            ok_detected = False
            thumb_up_detected = False

            if res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    if is_ok_gesture(hand_landmarks.landmark, w, h):
                        ok_detected = True
                    if is_thumb_up_gesture(hand_landmarks.landmark, w, h):
                        thumb_up_detected = True
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            now = time.time()

            # --- OK → OCR ---
            if ok_detected:
                if (now - last_print_ok) > cooldown_print_s:
                    print("ok")
                    last_print_ok = now
                if (now - last_ok_trigger) > cooldown_ocr_trigger_s:
                    last_ok_trigger = now
                    start_ocr_async(frame, show_window=False)

            # --- Thumb Up → YOLO ---
            if thumb_up_detected and (now - last_thumb_trigger) > cooldown_yolo_trigger_s:
                print("thumb up")
                last_thumb_trigger = now
                start_detection_async(frame, show_window=show_window)

            if show_window:
                cv2.putText(frame, "Press 'q' to quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if ok_detected:
                    cv2.putText(frame, "OK → OCR", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                if thumb_up_detected:
                    cv2.putText(frame, "THUMB UP → YOLO", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                cv2.imshow("Gestures: OK→OCR, ThumbUp→YOLO", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
