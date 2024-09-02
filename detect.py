from PIL import Image
from ultralytics import YOLO

# 单独推理
if __name__ == '__main__':
    # 加载模型
    model = YOLO("best.pt")

    # 从PIL图像进行预测
    im1 = Image.open("10001.png")
    results = model.predict(source=im1, save=True)  # 保存绘制后的图像

    # 遍历结果并打印每个检测到的目标的坐标
    for result in results:
        boxes = result.boxes  # 获取检测到的所有边界框

        for box in boxes:
            # 每个box包含坐标和其他信息
            x_min, y_min, x_max, y_max = box.xyxy[0]  # 获取左上角和右下角坐标
            print(f"Detected object at: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
