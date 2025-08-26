CameraWebServer.py streams what camera sees to an IP address, the IP address of raspberry pi. 

best_text_area.pt is a finetuned_yolo model particularly for text detection, currently 70% reliability.
I would further upgrade it for **better dection** and **object persistence**.

yolo_ocr_v2.py is the newest version combining text detection and OCR function. It combines every text area into a square-area image, OCR function, then print the interpreted result.


gestures_yolo_ocr.py, with gesture detection. An interactive way for user to give commands to glasses. 
1. "ok", OCR
2. "thumb up", object detection



**Pipeline**

1. Camera capture frame --> Gesture Detection --> Based on Result from Gesture Detection --> Send Result to API(chatgpt, gemini) --> Send Response to User --> Loop  (low latency, low power, low cost)
2. Camera capture frame --> Gesture Detection --> Based on Result from Gesture Detection --> Functions (OCR, Object Detection) --> Update Results on terminal, screen...etc --> Send Result to API(chatgpt, gemini) --> Send Response to User --> Loop
