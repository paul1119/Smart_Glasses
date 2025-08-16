import cv2
import time
import math
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def bbox_size(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return max(w, h)

def is_ok_gesture(landmarks, w, h):
    """
    Simple, robust-ish heuristic:
    1) Thumb tip and Index tip are very close (small vs hand bbox).
    2) Middle, Ring, Pinky are generally extended (tip higher than PIP in image coords).
       NOTE: 'higher' assumes an upright hand in the frame. Good enough for a demo.
    """
    # Convert normalized landmarks -> pixel coords
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]

    # Landmark indices (MediaPipe)
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18

    hand_scale = max(20.0, bbox_size(pts))  # prevent divide-by-small on tiny boxes
    thumb_index_dist = l2(pts[THUMB_TIP], pts[INDEX_TIP])

    # Condition 1: circle made by thumb & index
    close_enough = (thumb_index_dist / hand_scale) < 0.18  # tweakable threshold

    # Condition 2: other three fingers extended (tip above PIP => smaller y)
    middle_extended = pts[MIDDLE_TIP][1] < pts[MIDDLE_PIP][1]
    ring_extended   = pts[RING_TIP][1]   < pts[RING_PIP][1]
    pinky_extended  = pts[PINKY_TIP][1]  < pts[PINKY_PIP][1]

    return close_enough and middle_extended and ring_extended and pinky_extended

def main():
    cap = cv2.VideoCapture(0)  # try 0/1 depending on your camera index
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    # Draw window? Set to False if running headless
    show_window = False

    last_print = 0.0
    cooldown_s = 0.8  # prevent spamming "ok"

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,     # 0 is lighter for Pi
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to grab frame.")
                break

            # Flip for selfie-view; comment out if you prefer raw
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Mediapipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)

            ok_detected = False

            if res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    if is_ok_gesture(hand_landmarks.landmark, w, h):
                        ok_detected = True

                    # Optional drawing (costs a bit of CPU)
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            if ok_detected and (time.time() - last_print) > cooldown_s:
                print("ok")
                last_print = time.time()

            if show_window:
                cv2.putText(frame, "Press 'q' to quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if ok_detected:
                    cv2.putText(frame, "OK!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                cv2.imshow("OK Gesture Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
