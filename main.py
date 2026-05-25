import cv2
import mss as MSS
import time
import pydirectinput
import math
from ultralytics import YOLO

model = YOLO("best.pt").to("cuda")
cap = cv2.VideoCapture(1)

last_attack_time = 0

center_x = 320
center_y = 240

def center_in_box(center_x, center_y, box_x1, box_y1, box_x2, box_y2):
    global last_attack_time
    current_time = time.time()
    if not ((box_x1 <= center_x <= box_x2) and (box_y1 <= center_y <= box_y2)):
        return
    
    if current_time - last_attack_time < 0.1:
        return
    
    print("СТРЕЛЯЮ")
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
        
                center_in_box(center_x, center_y, x1, y1, x2, y2)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break