from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    result = model.train(data = r"D:\Minecraft players.v7i.yolov8\data.yaml", 
                     epochs = 100, 
                     imgsz = 640, 
                     device = "cuda")
