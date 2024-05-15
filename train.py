from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weight
model = YOLO('yolov8s.pt')

# Display model information (optional)
model.info()

# Train the model for 100 epochs
results = model.train(data='datasets/data.yaml', epochs=30, imgsz=1280, verbose=True, batch=8, device=[0,1,2,3])
