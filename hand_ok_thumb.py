import cv2
import time
import math
import mediapipe as mp

# --- OCR setup ---
import pytesseract

# --- YOLO setup ---
from ultralytics import YOLO

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# --- Utils ---
def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def bbox_size(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return max(w, h)

# --- Gesture: OK ---
def is_ok_gesture(landmarks, w, h):
    """
    Heuristic:
    (1) Thumb tip and Index tip are close (scaled by hand size).
    (2) Middle/Ring/Pinky are extended (tip above PIP in image coords).
    """
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]

    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP, MIDDLE_PIP = 12, 10
    RING_TIP,   RING_PIP   = 16, 14
    PINKY_TIP,  PINKY_PIP  = 20, 18

    hand_scale = max(20.0, bbox_size(pts))
    thumb_index_dist = l2(pts[THUMB_TIP], pts[INDEX_TIP])

    # Condition 1: circle made by thumb & index
    close_enough = (thumb_index_dist / hand_scale) < 0.18

    # Condition 2: other three fingers extended (smaller y is "up")
    middle_extended = pts[MIDDLE_TIP][1] < pts[MIDDLE_PIP][1]
    ring_extended   = pts[RING_TIP][1]   < pts[RING_PIP][1]
    pinky_extended  = pts[PINKY_TIP][1]  < pts[PINKY_PIP][1]

    return close_enough and middle_extended and ring_extended and pinky_extended

# --- Gesture: Thumbs Up ---
def is_thumb_up_gesture(landmarks, w, h):
    """
    Heuristic:
    - Thumb extended upward: vector (MCP->TIP) points mostly up and is long enough.
    - Other four fingers folded (tip closer to wrist than PIP).
    - Thumb not touching index tip (avoid OK confusion).
    """
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]

    WRIST = 0
    THUMB_MCP, THUMB_TIP = 2, 4
    INDEX_PIP, INDEX_TIP = 6, 8
    MIDDLE_PIP, MIDDLE_TIP = 10, 12
    RING_PIP, RING_TIP = 14, 16
    PINKY_PIP, PINKY_TIP = 18, 20

    hand_scale = max(20.0, bbox_size(pts))

    # 1) Thumb vector mostly up and long enough (negative y is up in image)
    v = (pts[THUMB_TIP][0] - pts[THUMB_MCP][0], pts[THUMB_TIP][1] - pts[THUMB_MCP][1])
    v_len = max(1e-6, math.hypot(v[0], v[1]))
    cos_to_up = -(v[1]) / v_len           # dot with unit up (0,-1)
    thumb_long_enough = (v_len / hand_scale) > 0.25
    thumb_up = (cos_to_up > 0.6) and thumb_long_enough

    # 2) Other fingers folded: tip closer to wrist than PIP
    wr = pts[WRIST]
    def folded(tip, pip):
        return l2(pts[tip], wr) < 0.9 * l2(pts[pip], wr)

    others_folded = all([
        folded(INDEX_TIP, INDEX_PIP),
        folded(MIDDLE_TIP, MIDDLE_PIP),
        folded(RING_TIP,   RING_PIP),
        folded(PINKY_TIP,  PINKY_PIP),
    ])

    # 3) Not OK-like: keep thumb & index tip reasonably apart
    not_ok_like = (l2(pts[THUMB_TIP], pts[INDEX_TIP]) / hand_scale) > 0.22

    return thumb_up and others_folded and not_ok_like

# --- OCR function ---
def ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    config = '--oem 1 --psm 3'
    text = pytesseract.image_to_string(thresh, config=config)
    return text

# --- YOLO object detection ---
def object_detection(frame):
    model = YOLO('yolov8n.pt')
    results = model(frame, verbose = False)
    r = results[0]

    # Print detections
    for b in r.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        print(f"[YOLO] {r.names[cls_id]} {conf:.2f} @ ({x1},{y1},{x2},{y2})")


def main():
    cap = cv2.VideoCapture(0)  # adjust index if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    show_window = False  # set False if running headless

    last_print_ok = 0.0
    last_print_thumb = 0.0
    cooldown_s = 0.8

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,       # lighter for Raspberry Pi
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)  # selfie view
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

                    # Optional: draw landmarks (costs some CPU)
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            now = time.time()
            if ok_detected and (now - last_print_ok) > cooldown_s:
                print("ok")
                ocr(frame)
                last_print_ok = now

            if thumb_up_detected and (now - last_print_thumb) > cooldown_s:
                print("thumb up")
                object_detection(frame)
                last_print_thumb = now


            if show_window:
                cv2.putText(frame, "Press 'q' to quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if ok_detected:
                    cv2.putText(frame, "OK!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                if thumb_up_detected:
                    cv2.putText(frame, "THUMB UP!", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                cv2.imshow("OK & Thumbs Up Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
