from ultralytics import YOLO

# 训练
if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data='./dataset/data.yaml', epochs=300, imgsz=320)
