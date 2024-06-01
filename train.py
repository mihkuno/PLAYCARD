from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weight
# model = YOLO('/home/justeengg/app/runs/detect/train2/weights/best.pt')
model = YOLO('yolov8n.pt')

# Display model information (optional)
model.info()

# Train the model for 100 epochs
results = model.train(data='/home/justeengg/app/datasets/data.yaml', epochs=100, imgsz=2000, verbose=True, batch=12, device=[0,1,2,3], workers=46)

