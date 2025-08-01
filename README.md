CameraWebServer.py streams what camera sees to an IP address, the IP address of raspberry pi. 

best_text_area.pt is a finetuned_yolo model particularly for text detection, currently 70% reliability.
I would further upgrade it for **better dection** and **object persistence**.

yolo_ocr_v2.py is the newest version combining text detection and OCR function. It combines every text area into a square-area image, OCR function, then print the interpreted result.
