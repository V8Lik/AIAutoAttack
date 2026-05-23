import cv2
import mss as MSS
import time
import numpy as np
import pydirectinput
from ultralytics import YOLO

model = YOLO("best.pt").to("cuda")
cap = cv2.VideoCapture(1)

last_attack_time = 0

def trigger_attack(x1, y1, x2, y2):
    global last_attack_time
    current_time = time.time()
    if x1 >= 300 and x2 <= 340 and y1 >= 220 and y2 <= 260:
        if current_time - last_attack_time >= 2:
            pydirectinput.leftClick()
            last_attack_time = current_time

with MSS.mss() as sct:
    while True:
        last_time = time.time()
        
        ret, frame = cap.read()
        results = model.predict(frame)
        bbox_frame = results[0].plot()
        
        cv2.imshow("Test Cupture", bbox_frame)
        
        print(f"fps: {1 / (time.time() - last_time)}")

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
                print(f"Координаты: {x1}, {y1}, {x2}, {y2}")
        
                trigger_attack(x1, y1, x2, y2)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break