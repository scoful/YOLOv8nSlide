from ultralytics import YOLO

# 批量验证
if __name__ == '__main__':
    # Load a model
    model = YOLO('best.pt')  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
