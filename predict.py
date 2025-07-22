# from cv2 import cv2
import cv2
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
# import cv2
from torchvision.transforms import ToTensor
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading

from PLNnet import pretrained_inception
# from new_resnet import pretrained_inception

# from draw_rectangle import draw
from PLNnet import *

# voc数据集的类别信息，这里转换成字典形式
classes = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5, "car": 6, "cat": 7, "chair": 8,
           "cow": 9, "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15,
           "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19}

# 类别信息，这里写的都是voc数据集的，如果是自己的数据集需要更改
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
# 类别总数：20
CLASS_NUM = len(VOC_CLASSES)

# 图像预处理均值
MEAN = np.array((123, 117, 104), dtype=np.float32)  # RGB

def init_model(weights_path=r"E:\study\2025spring\baoyan\Paper reproduction\PLN\PLN\results\5\pln_latest.pth"):
    # 使用混合精度训练加速
    if torch.cuda.is_available():
        # 设置GPU内存分配策略，避免内存碎片
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True  # 使用cudnn加速卷积操作
    
    # 创建模型
    model = inceptionresnetv2(num_classes=20, pretrained='imagenet').cuda()
    
    # 加载权重
    checkpoint = torch.load(weights_path, map_location="cuda")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    epoch = checkpoint['epoch']
    print("Epoch", epoch)
    
    # 设置为测试模式
    model.eval()
    
    return model

def preprocess_image(img_path):
    """图像预处理函数"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        return None, None
        
    # 保留原始图像用于绘制
    orig_img = img.copy()
    
    # 调整图像大小
    img = cv2.resize(img, (448, 448))
    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 减去均值
    img = img - MEAN
    # 转为tensor
    img = ToTensor()(img)
    # 添加batch维度
    img = img.unsqueeze(0)
    
    return img, orig_img

def preprocess_images_batch(img_paths, target_size=(448, 448)):
    """批量预处理图像函数"""
    processed_images = []
    original_images = []
    valid_paths = []
    
    # 遍历所有图像路径
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 保存原始图像
        original_images.append(img.copy())
        valid_paths.append(img_path)
        
        # 预处理图像
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img - MEAN
        img = ToTensor()(img)
        processed_images.append(img)
    
    # 如果没有有效图像，返回空结果
    if not processed_images:
        return None, None, None
    
    # 将预处理后的图像堆叠成批处理张量
    batch_tensor = torch.stack(processed_images)
    
    return batch_tensor, original_images, valid_paths

class Pred():
    # 参数初始化
    def __init__(self, model, img_root=None, score_confident=0.1, p_confident=0.1, nms_confident=0.1, iou_con=0.5,
                 output_dir="output", batch_size=4, num_workers=4, min_size=200, aspect_ratio_threshold=5.0,
                 overlap_threshold=0.7, center_dist_threshold=0.3, area_ratio_threshold=0.5):
        self.model = model
        self.img_root = img_root
        self.score_confident = score_confident
        self.p_confident = p_confident
        self.nms_confident = nms_confident
        self.iou_con = iou_con
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lock = threading.Lock()  # 用于线程安全的打印
        # 新增参数
        self.min_size = min_size  # 检测框的最小面积
        self.aspect_ratio_threshold = aspect_ratio_threshold  # 长宽比阈值，超过此值的框会被抑制
        # 增加重叠框检测参数
        self.overlap_threshold = overlap_threshold  # 重叠框判定阈值
        self.center_dist_threshold = center_dist_threshold  # 中心点距离阈值
        self.area_ratio_threshold = area_ratio_threshold  # 面积比例阈值
        
        # 添加类别信息
        self.voc_classes = VOC_CLASSES
        self.classes = {class_name: i for i, class_name in enumerate(VOC_CLASSES)}
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def set_thresholds(self, score_confident=None, p_confident=None, nms_confident=None, iou_con=None, 
                        min_size=None, aspect_ratio_threshold=None, overlap_threshold=None,
                        center_dist_threshold=None, area_ratio_threshold=None):
        """设置预测阈值"""
        if score_confident is not None:
            self.score_confident = score_confident
        if p_confident is not None:
            self.p_confident = p_confident
        if nms_confident is not None:
            self.nms_confident = nms_confident
        if iou_con is not None:
            self.iou_con = iou_con
        if min_size is not None:
            self.min_size = min_size
        if aspect_ratio_threshold is not None:
            self.aspect_ratio_threshold = aspect_ratio_threshold
        if overlap_threshold is not None:
            self.overlap_threshold = overlap_threshold
        if center_dist_threshold is not None:
            self.center_dist_threshold = center_dist_threshold
        if area_ratio_threshold is not None:
            self.area_ratio_threshold = area_ratio_threshold
        
        print(f"当前阈值设置: score={self.score_confident}, p={self.p_confident}, nms={self.nms_confident}, iou={self.iou_con}")
        print(f"检测框过滤设置: 最小面积={self.min_size}, 长宽比阈值={self.aspect_ratio_threshold}")
        print(f"重叠框抑制设置: 重叠阈值={self.overlap_threshold}, 中心距离阈值={self.center_dist_threshold}, 面积比阈值={self.area_ratio_threshold}")
    
    def compute_area(self, branch, j, i):
        """使用缓存版本的计算区域函数"""
        area = [[],[]]
        if branch == 0:
            # 左下角
            area = [[0, j+1],[i, 14]]
        elif branch == 1:
            # 左上角
            area = [[0, j+1],[0, i+1]]
        elif branch == 2:
            # 右下角
            area = [[j, 14],[i, 14]]
        elif branch == 3:
            # 右上角
            area = [[j, 14],[0, i+1]]
        return area

    def predict_folder(self, folder_path):
        """预测文件夹中的所有图像（使用优化的批处理方式）"""
        # 支持的图像格式
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # 确保文件夹存在
        if not os.path.exists(folder_path):
            print(f"文件夹 {folder_path} 不存在!")
            return
        
        # 获取文件夹中的所有图像文件
        img_files = []
        for file_name in os.listdir(folder_path):
            ext = os.path.splitext(file_name)[1].lower()
            if ext in img_extensions:
                img_files.append(os.path.join(folder_path, file_name))
        
        if not img_files:
            print(f"文件夹 {folder_path} 中没有找到支持的图像文件!")
            return
        
        # 预热CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 批处理图像
        batches = [img_files[i:i + self.batch_size] for i in range(0, len(img_files), self.batch_size)]
        
        print(f"找到 {len(img_files)} 个图像文件，分为 {len(batches)} 个批次进行处理...")
        start_time = time.time()
        
        # 处理所有批次
        total_images = 0
        for batch_idx, batch_paths in enumerate(batches):
            print(f"处理批次 {batch_idx+1}/{len(batches)}，包含 {len(batch_paths)} 张图像")
            total_images += self.process_batch_gpu(batch_paths, batch_idx, len(batches))
        
        elapsed = time.time() - start_time
        print(f"预测完成! 总共成功处理 {total_images} 张图像，总共耗时 {elapsed:.2f} 秒，"
              f"平均每张图像 {elapsed/total_images:.4f} 秒")
        print(f"结果保存在 {self.output_dir} 文件夹中")

    def process_batch_gpu(self, img_paths, batch_idx, total_batches):
        """使用GPU批处理图像"""
        # 批量预处理图像
        batch_tensor, original_images, valid_paths = preprocess_images_batch(img_paths)
        
        # 如果没有有效图像，返回0
        if batch_tensor is None:
            print(f"批次 {batch_idx+1}/{total_batches} 中没有有效图像")
            return 0
        
        # 将图像批次传入GPU
        batch_tensor = batch_tensor.to("cuda")
        
        # 批量推理
        with torch.no_grad():
            results = self.model(batch_tensor)
        
        # 处理每张图像的结果
        processed_count = 0
        for i in range(len(valid_paths)):
            try:
                # 为每张图像单独处理结果
                img_results = []
                for branch in range(4):
                    # 从批处理结果中提取单个图像的分支结果
                    branch_result = results[branch][i].permute(1, 2, 0)
                    # 解码处理
                    bbox = self.Decode(branch, branch_result.clone() if branch in [1, 2] else branch_result)
                    img_results.append(bbox)
                
                # 处理单张图像的检测结果
                self.process_single_result(img_results, original_images[i], valid_paths[i])
                processed_count += 1
                
                print(f"  - 成功处理: {os.path.basename(valid_paths[i])}")
            except Exception as e:
                print(f"  - 处理失败: {os.path.basename(valid_paths[i])}, 错误: {e}")
        
        return processed_count

    def process_single_result(self, results, original_img, img_path):
        """处理单个图像的检测结果"""
        # 合并四个分支的结果
        bbox = torch.cat(results, dim=0)
        
        # 预处理过滤
        if bbox.shape[0] > 0:
            # 使用向量化操作进行过滤，提高效率
            # 1. 过滤置信度低的框
            conf_mask = bbox[:, 4] >= self.nms_confident
            bbox = bbox[conf_mask]
            
            # 2. 过滤无效框
            if bbox.shape[0] > 0:
                # 计算所有框的宽度和高度
                widths = bbox[:, 2] - bbox[:, 0]
                heights = bbox[:, 3] - bbox[:, 1]
                areas = widths * heights
                
                # 检查面积
                area_mask = areas >= self.min_size
                
                # 检查长宽比
                aspect_ratios = torch.max(widths / (heights + 1e-6), heights / (widths + 1e-6))
                ratio_mask = aspect_ratios <= self.aspect_ratio_threshold
                
                # 组合掩码
                valid_mask = area_mask & ratio_mask
                bbox = bbox[valid_mask]
        
        # 非极大值抑制处理
        bboxes = self.NMS(bbox)
        
        # 调整图像大小用于绘制
        output_img = cv2.resize(original_img, (448, 448))
        
        # 如果没有检测到物体
        if len(bboxes) == 0:
            file_name = os.path.basename(img_path)
            base_name, ext = os.path.splitext(file_name)
            output_path = os.path.join(self.output_dir, f"{base_name}{ext}")
            cv2.imwrite(output_path, output_img)
            return []
        
        # 绘制检测结果
        detection_results = []
        for i in range(len(bboxes)):
            x1 = bboxes[i][0].item()
            y1 = bboxes[i][1].item()
            x2 = bboxes[i][2].item()
            y2 = bboxes[i][3].item()
            score = bboxes[i][4].item()
            class_name = bboxes[i][5].item()
            class_label = self.voc_classes[int(class_name)]
            
            detection_results.append({
                'class': class_label,
                'score': score,
                'bbox': [x1, y1, x2, y2]
            })
            
            text = class_label + '{:.3f}'.format(score)
            cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(output_img, text, (int(x1) + 10, int(y1) + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        
        # 保存结果图像
        file_name = os.path.basename(img_path)
        base_name, ext = os.path.splitext(file_name)
        output_path = os.path.join(self.output_dir, f"{base_name}{ext}")
        cv2.imwrite(output_path, output_img)
        
        return detection_results

    # 接受的result的形状为1*7*7*204
    def Decode(self, branch, result):
        result = result.squeeze()
        # result [14*14*204]
        # 0 pij 1x 2y
        # 3-16 lxij 16-30 lyij
        # 31-50 qij
        r = []
        bboxes_ = list()
        labels_ = list()
        scores_ = list()
        
        # 使用向量化操作预先计算softmax
        for i in range(14):
            for j in range(14):
                for p in range(4):
                    result[i, j, 51 * p + 3:51 * p + 17] = torch.softmax(result[i, j, 51 * p + 3:51 * p + 17], dim=-1)
                    result[i, j, 51 * p + 17:51 * p + 31] = torch.softmax(result[i, j, 51 * p + 17:51 * p + 31], dim=-1)
                    result[i, j, 51 * p + 31:51 * p + 51] = torch.softmax(result[i, j, 51 * p + 31:51 * p + 51], dim=-1)

        for p in range(2):
            # ij center || mn corner
            for i in range(14):
                for j in range(14):
                    if result[i, j, p * 51] < self.p_confident:
                        continue
                    
                    # 获取区域
                    x_area, y_area = self.compute_area(branch, j, i)
                    
                    # 提取当前单元格的值，避免重复访问
                    p_ij = result[i, j, 51 * p + 0]
                    i_ = result[i, j, 51 * p + 2]
                    j_ = result[i, j, 51 * p + 1]
                    
                    # 预先获取所有类别的q_cij值
                    q_cijs = result[i, j, 51 * p + 31:51 * p + 51]
                    
                    for n in range(y_area[0], y_area[1]):
                        for m in range(x_area[0], x_area[1]):
                            # 提取当前角落单元格的值
                            p_nm = result[n, m, 51 * (p + 2) + 0]
                            n_ = result[n, m, 51 * (p + 2) + 2]
                            m_ = result[n, m, 51 * (p + 2) + 1]
                            
                            # 计算交互项
                            l_ij_x = result[i, j, 51 * p + 3 + m]
                            l_ij_y = result[i, j, 51 * p + 3 + n]
                            l_nm_x = result[n, m, 51 * (p + 2) + 17 + j]
                            l_nm_y = result[n, m, 51 * (p + 2) + 17 + i]
                            
                            # 计算共同项
                            common_factor = p_ij * p_nm * (l_ij_x * l_ij_y + l_nm_x * l_nm_y) / 2 * 1000
                            
                            # 获取所有类别的q_cnm值
                            q_cnms = result[n, m, 51 * (p + 2) + 31:51 * (p + 2) + 51]
                            
                            # 计算所有类别的分数
                            scores = common_factor * q_cijs * q_cnms
                            
                            # 找出超过阈值的类别
                            for c in range(20):
                                score = scores[c].item()
                                if score > self.score_confident:
                                    r.append([i + i_, j + j_, n + n_, m + m_, c, score])
            
            # 处理识别结果
            for l in r:
                # 重新encode 变为xmin,ymin,xmax,ymax,score.class
                if branch == 0:
                    # 左下角
                    bbox = [l[3], 2 * l[0] - l[2], 2 * l[1] - l[3], l[2]]
                elif branch == 1:
                    # 左上角
                    bbox = [l[3], l[2], 2 * l[1] - l[3], 2 * l[0] - l[2]]
                elif branch == 2:
                    # 右下角
                    bbox = [2 * l[1] - l[3], 2 * l[0] - l[2], l[3], l[2]]
                elif branch == 3:
                    # 右上角
                    bbox = [2 * l[1] - l[3], l[2], l[3], 2 * l[0] - l[2]]

                # 缩放
                bbox = [b * 32 for b in bbox]
                bboxes_.append(bbox)
                labels_.append(l[4])
                scores_.append(l[5])

        # 创建tensor并填充数据
        if not labels_:
            return torch.zeros((0, 6), device="cuda")
            
        bbox_info = torch.zeros(len(labels_), 6, device="cuda")
        for i in range(len(labels_)):
            bbox_info[i, 0] = bboxes_[i][0]
            bbox_info[i, 1] = bboxes_[i][1]
            bbox_info[i, 2] = bboxes_[i][2]
            bbox_info[i, 3] = bboxes_[i][3]
            bbox_info[i, 4] = scores_[i]
            bbox_info[i, 5] = labels_[i]

        return bbox_info
    
    def _is_valid_box(self, box):
        """检查边界框是否有效（面积大小和长宽比例）"""
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        
        # 宽度和高度
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        
        # 检查面积
        area = width * height
        if area < self.min_size:
            return False
        
        # 检查长宽比例
        if width > 0 and height > 0:
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > self.aspect_ratio_threshold:
                return False
                
        return True

    def NMS(self, bbox):
        """非极大值抑制处理（优化版本）"""
        if bbox.shape[0] == 0:
            return []
            
        # 存放最终需要保留的预测框
        bboxes = []
        
        # 取出每个grid cell中的类别信息
        ori_class_index = bbox[:, 5]
        
        # 按照类别进行排序
        class_index, class_order = ori_class_index.sort(dim=0, descending=False)
        class_index = class_index.tolist()
        
        # 根据排序后的索引更改bbox排列顺序
        bbox = bbox[class_order, :]
        
        a = 0
        # 对每个类别分别进行NMS
        for i in range(CLASS_NUM):
            # 统计该类别目标数量
            num = class_index.count(i)
            if num == 0:
                continue
                
            # 提取同一类别的所有信息
            x = bbox[a:a + num, :]
            
            # 提取概率信息并排序
            score = x[:, 4]
            _, score_order = score.sort(dim=0, descending=True)
            y = x[score_order, :]
            
            # 检查最高概率是否满足阈值
            if y[0, 4] >= self.nms_confident:
                # 仅处理达到阈值的检测框
                confidence_mask = y[:, 4] >= self.nms_confident
                y_filtered = y[confidence_mask]
                
                if y_filtered.shape[0] == 0:
                    a += num
                    continue
                
                # 按置信度降序排列
                scores = y_filtered[:, 4]
                _, order = scores.sort(dim=0, descending=True)
                boxes = y_filtered[order]
                
                # 计算所有框的面积
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                
                # 检查框是否有效（面积和长宽比）
                valid_mask = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)
                
                # 宽度和高度
                width = boxes[:, 2] - boxes[:, 0]
                height = boxes[:, 3] - boxes[:, 1]
                
                # 过滤无效框
                for j in range(boxes.shape[0]):
                    # 检查面积
                    if area[j] < self.min_size:
                        valid_mask[j] = False
                        continue
                    
                    # 检查长宽比
                    if width[j] > 0 and height[j] > 0:
                        aspect_ratio = max(width[j] / height[j], height[j] / width[j])
                        if aspect_ratio > self.aspect_ratio_threshold:
                            valid_mask[j] = False
                
                # 应用有效性掩码
                boxes = boxes[valid_mask]
                area = area[valid_mask]
                
                # 如果没有有效框，跳过
                if boxes.shape[0] == 0:
                    a += num
                    continue
                
                # 使用向量化操作进行NMS
                keep = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)
                
                for j in range(boxes.shape[0] - 1):
                    if not keep[j]:
                        continue
                    
                    # 获取当前框
                    box_j = boxes[j, :4]
                    
                    # 计算当前框与所有其他框的IoU
                    # 获取所有剩余框
                    remaining_boxes = boxes[j+1:, :4]
                    
                    # 计算交集的左上角和右下角
                    xx1 = torch.max(box_j[0], remaining_boxes[:, 0])
                    yy1 = torch.max(box_j[1], remaining_boxes[:, 1])
                    xx2 = torch.min(box_j[2], remaining_boxes[:, 2])
                    yy2 = torch.min(box_j[3], remaining_boxes[:, 3])
                    
                    # 计算交集的宽和高
                    w = torch.clamp(xx2 - xx1, min=0)
                    h = torch.clamp(yy2 - yy1, min=0)
                    
                    # 计算交集面积
                    inter = w * h
                    
                    # 计算并集面积
                    area_j = area[j]
                    area_remaining = area[j+1:]
                    union = area_j + area_remaining - inter
                    
                    # 计算IoU
                    iou = inter / union
                    
                    # 计算中心点
                    # 计算中心点坐标
                    box_j_center_x = (box_j[0] + box_j[2]) / 2
                    box_j_center_y = (box_j[1] + box_j[3]) / 2
                    
                    remaining_centers_x = (remaining_boxes[:, 0] + remaining_boxes[:, 2]) / 2
                    remaining_centers_y = (remaining_boxes[:, 1] + remaining_boxes[:, 3]) / 2
                    
                    # 计算距离平方
                    dist_x = (box_j_center_x - remaining_centers_x)
                    dist_y = (box_j_center_y - remaining_centers_y)
                    
                    # 计算最大宽高
                    width_j = box_j[2] - box_j[0]
                    height_j = box_j[3] - box_j[1]
                    
                    width_remaining = remaining_boxes[:, 2] - remaining_boxes[:, 0]
                    height_remaining = remaining_boxes[:, 3] - remaining_boxes[:, 1]
                    
                    max_width = torch.max(width_j.expand_as(width_remaining), width_remaining)
                    max_height = torch.max(height_j.expand_as(height_remaining), height_remaining)
                    
                    # 归一化距离
                    norm_dist_x = dist_x / (max_width + 1e-6)
                    norm_dist_y = dist_y / (max_height + 1e-6)
                    
                    # 计算中心点距离
                    center_dist = torch.sqrt(norm_dist_x**2 + norm_dist_y**2)
                    
                    # 面积比
                    min_area = torch.min(area_j.expand_as(area_remaining), area_remaining)
                    max_area = torch.max(area_j.expand_as(area_remaining), area_remaining)
                    area_ratio = min_area / (max_area + 1e-6)
                    
                    # 检查重叠
                    overlap_ratio = inter / (min_area + 1e-6)
                    contains = overlap_ratio > self.overlap_threshold
                    
                    # 应用NMS规则
                    # 1. IoU阈值
                    # 2. 中心点距离小且面积比小
                    # 3. 包含关系
                    to_suppress = (iou >= self.iou_con) | \
                                 ((center_dist < self.center_dist_threshold) & (area_ratio < self.area_ratio_threshold)) | \
                                 contains
                    
                    # 标记需要抑制的框
                    keep[j+1:][to_suppress] = False
                
                # 收集所有保留的框
                for idx in range(boxes.shape[0]):
                    if keep[idx]:
                        bboxes.append(boxes[idx])
                
            # 更新索引
            a += num
            
        return bboxes

    def result(self, img_path=None):
        """处理单张图像并返回检测结果（兼容旧接口）"""
        # 使用传入的图像路径或实例变量
        img_path = img_path if img_path is not None else self.img_root
        
        # 预处理图像
        img_tensor, original_img = preprocess_image(img_path)
        if img_tensor is None:
            print(f"无法读取图像: {img_path}")
            return []
        
        # 添加批次维度
        img_tensor = img_tensor.unsqueeze(0)
        
        # 将tensor移到GPU
        img_tensor = img_tensor.to("cuda")
        
        # 使用优化的批处理处理流程
        with torch.no_grad():
            results = self.model(img_tensor)
        
        # 为单张图像处理结果
        img_results = []
        for branch in range(4):
            # 从批处理结果中提取单个图像的分支结果
            branch_result = results[branch][0].permute(1, 2, 0)
            # 解码处理
            bbox = self.Decode(branch, branch_result.clone() if branch in [1, 2] else branch_result)
            img_results.append(bbox)
        
        # 处理单张图像的检测结果
        return self.process_single_result(img_results, original_img, img_path)


def parse_args():
    parser = argparse.ArgumentParser(description='PLN图像目标检测预测脚本')
    parser.add_argument('--folder', type=str, default="test_images", help='要预测的图像文件夹路径')
    parser.add_argument('--image', type=str, default=None, help='要预测的单张图像路径')
    parser.add_argument('--weights', type=str, default=r"E:\study\2025spring\baoyan\Paper reproduction\PLN\PLN\results\6\pln_latest.pth",
                        help='模型权重文件路径')
    parser.add_argument('--output_dir', type=str, default='output4', help='输出结果保存目录')
    parser.add_argument('--score_threshold', type=float, default=0.1, help='得分阈值')
    parser.add_argument('--p_threshold', type=float, default=0.1, help='p阈值')
    parser.add_argument('--nms_threshold', type=float, default=0.1, help='NMS阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.25, help='IOU阈值')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=8, help='工作线程数')
    parser.add_argument('--min_size', type=float, default=800, help='检测框最小面积')
    parser.add_argument('--aspect_ratio_threshold', type=float, default=3.5, help='长宽比例阈值，超过此值的框会被抑制')
    parser.add_argument('--overlap_threshold', type=float, default=1, help='重叠框判定阈值，两框重叠超过此比例的较小框面积时会被抑制')
    parser.add_argument('--center_dist_threshold', type=float, default=0.3, help='中心点距离阈值，归一化距离小于此值时触发抑制')
    parser.add_argument('--area_ratio_threshold', type=float, default=0.5, help='面积比例阈值，小于此值时小框可能被抑制')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    start_time = time.time()
    
    # 初始化模型
    model = init_model(args.weights)
    
    # 创建预测器实例
    predictor = Pred(
        model=model,
        score_confident=args.score_threshold,
        p_confident=args.p_threshold,
        nms_confident=args.nms_threshold,
        iou_con=args.iou_threshold,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_size=args.min_size,
        aspect_ratio_threshold=args.aspect_ratio_threshold,
        overlap_threshold=args.overlap_threshold,
        center_dist_threshold=args.center_dist_threshold,
        area_ratio_threshold=args.area_ratio_threshold
    )
    
    # 预测单张图像或文件夹
    if args.folder:
        predictor.predict_folder(args.folder)
    elif args.image:
        predictor.img_root = args.image
        predictor.result(args.image)
    else:
        print("请指定要预测的图像文件夹(--folder)或单张图像路径(--image)")
    
    total_time = time.time() - start_time
    print(f"总运行时间: {total_time:.2f} 秒")
