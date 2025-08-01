from flask import Flask, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def get_frame():
    success, frame = camera.read()
    if not success:
        return None
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

@app.route('/video')
def video():
    def generate_frames():
        while True:
            frame = get_frame()
            if frame is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    frame = get_frame()
    if frame is None:
        return "Camera error", 500
    return Response(frame, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
