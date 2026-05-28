from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    result = model.train(data = r"\data.yaml", 
                     epochs = 100, 
                     imgsz = 640, 
                     device = "cuda")