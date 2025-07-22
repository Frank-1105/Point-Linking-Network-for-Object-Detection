import cv2
import os
import numpy as np
from pathlib import Path

# VOC数据集的类别信息
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

# 为每个类别随机生成一个颜色
COLOR_MAP = {
    i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(VOC_CLASSES))
}


def draw_boxes(image, boxes, line_thickness=2, font_scale=0.75):
    """
    在图像上绘制边界框

    参数:
        image: 要绘制的图像
        boxes: 边界框列表，每个框的格式为 [x_min, y_min, x_max, y_max, class_id]
        line_thickness: 线条粗细
        font_scale: 字体大小
    """
    for box in boxes:
        x1, y1, x2, y2, class_id = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # 获取类别颜色
        color = COLOR_MAP[class_id]

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # 绘制类别标签
        label = VOC_CLASSES[class_id]
        cv2.putText(image, label, (x1, y1 +15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, line_thickness)


def process_image(image_path, boxes, output_dir):
    """
    处理单张图片及其标注

    参数:
        image_path: 图片路径
        boxes: 标注框列表
        output_dir: 输出目录
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    # 绘制边界框
    draw_boxes(image, boxes)

    # 保存结果
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"已保存结果到: {output_path}")


def process_directory(image_dir, annotation_file, output_dir):
    """
    处理整个目录的图片和标注

    参数:
        image_dir: 图片目录
        annotation_file: 大的标注文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取标注文件
    annotation_dict = {}
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            image_name = parts[0]
            num_boxes = (len(parts) - 1) // 5
            boxes = []
            for i in range(num_boxes):
                xmin = float(parts[i * 5 + 1])
                ymin = float(parts[i * 5 + 2])
                xmax = float(parts[i * 5 + 3])
                ymax = float(parts[i * 5 + 4])
                class_id = int(parts[i * 5 + 5])
                boxes.append([xmin, ymin, xmax, ymax, class_id])
            annotation_dict[image_name] = boxes

    # 获取所有图片文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))

    # 处理每张图片
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        if image_name in annotation_dict:
            print(f"处理图片: {image_path}")
            process_image(str(image_path), annotation_dict[image_name], output_dir)
        else:
            print(f"未找到图片 {image_name} 的标注信息")


def main():
    # 配置路径
    image_dir = r"C:\Users\ZHENGZHIQIAN\Desktop\PLN\output1"  # 图片目录
    annotation_file = r"C:\Users\ZHENGZHIQIAN\Desktop\PLN\vocall448.txt"  # 大的标注文件路径
    output_dir = r"C:\Users\ZHENGZHIQIAN\Desktop\PLN\outputgt"  # 输出目录

    # 处理整个目录
    process_directory(image_dir, annotation_file, output_dir)


if __name__ == "__main__":
    main()
