import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
import time
import json
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from collections import defaultdict
import matplotlib.pyplot as plt

# 导入项目模块
from PLNdata import PLNDataset
from PLNLoss import PLNLoss
from PLNnet import pretrained_inception
from predict import Pred  # 导入预测类


# VOC类别信息
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
CLASS_NUM = len(VOC_CLASSES)


class EvalDataset(Dataset):
    """用于评估的数据集类"""
    def __init__(self, image_paths, transforms=None, test_file=None):
        self.image_paths = image_paths
        self.transforms = transforms
        self.test_file = test_file
        self.ground_truth_boxes = None
        
        # 如果提供了测试文件，提前提取所有真实标签
        if self.test_file and os.path.exists(self.test_file):
            self.ground_truth_boxes = self._extract_all_boxes()
            print(f"预加载了 {len(self.ground_truth_boxes)} 个标注框")

    def _extract_all_boxes(self):
        """从测试文件中提取所有边界框"""
        if not self.test_file:
            return []
            
        try:
            return extract_boxes_from_targets(self.test_file, 0, float('inf'))
        except Exception as e:
            print(f"提取边界框失败: {e}")
            return []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            # 如果图像无法读取，返回一个空图像
            img = np.zeros((448, 448, 3), dtype=np.uint8)
        
        # 预处理图像
        img = cv2.resize(img, (448, 448))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 减去均值
        img = img - np.array((123, 117, 104), dtype=np.float32)
        # 转为tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        return img, img_path


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU

    参数:
        box1: [x1, y1, x2, y2] 格式的边界框
        box2: [x1, y1, x2, y2] 格式的边界框

    返回:
        IoU值
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    intersection_area = max(0, (x2 - x1) ) * max(0, (y2 - y1) )

    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0] ) * (box1[3] - box1[1] )
    box2_area = (box2[2] - box2[0] ) * (box2[3] - box2[1] )

    # 计算并集面积和IoU
    union_area = box1_area + box2_area - intersection_area

    # 防止除0错误

    iou = intersection_area / union_area

    return iou

def extract_boxes_from_targets(file_path, start_index, end_index):
    gt_boxes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if end_index == float('inf'):
                end_index = len(lines)
            for line in lines[start_index:end_index]:
                parts = line.strip().split()
                # 忽略图片名，直接处理边界框信息
                box_parts = parts[1:]
                num_boxes = len(box_parts) // 5
                for i in range(num_boxes):
                    # 提取边界框坐标和类别信息
                    xmin = float(box_parts[i * 5])
                    ymin = float(box_parts[i * 5 + 1])
                    xmax = float(box_parts[i * 5 + 2])
                    ymax = float(box_parts[i * 5 + 3])
                    class_id = int(box_parts[i * 5 + 4])
                    # 构建单个边界框列表
                    gt_box = [xmin, ymin, xmax, ymax, 1.0, class_id]
                    # 添加到结果列表
                    gt_boxes.append(gt_box)
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。")
    except Exception as e:
        print(f"发生未知错误：{e}")
    return gt_boxes


def calculate_ap_per_class(detections, ground_truths, class_idx, iou_threshold=0.5):
    """
    计算特定类别的平均精度(AP)
    
    参数:
        detections: 检测结果列表 [[x1,y1,x2,y2,score,class_id], ...]
        ground_truths: 真实标注列表 [[x1,y1,x2,y2,score,class_id], ...]
        class_idx: 当前类别索引
        iou_threshold: IoU阈值
        
    返回:
        ap: 该类别的平均精度
        precision: 精度数组
        recall: 召回率数组
    """
    # 过滤当前类别的预测和真实标注
    class_dets = [d for d in detections if d[5] == class_idx]
    class_gt = [g for g in ground_truths if g[5] == class_idx]
    
    # 如果没有真实标注，AP为0
    if len(class_gt) == 0:
        return 0.0, np.array([]), np.array([])
    
    # 按置信度排序预测结果
    class_dets = sorted(class_dets, key=lambda x: x[4], reverse=True)
    
    # 初始化TP和FP数组
    tp = np.zeros(len(class_dets))
    fp = np.zeros(len(class_dets))
    
    # 标记已匹配的真实标注
    gt_matched = [False] * len(class_gt)
    
    # 遍历每个检测结果
    for i, det in enumerate(class_dets):
        # 找到与当前检测结果IoU最大的真实标注
        max_iou = -float('inf')
        max_idx = -1
        
        for j, gt in enumerate(class_gt):
            # 如果该真实标注已经被匹配，跳过
            if gt_matched[j]:
                continue
                
            # 计算IoU
            iou = calculate_iou(det[:4], gt[:4])
            
            # 更新最大IoU
            if iou > max_iou:
                max_iou = iou
                max_idx = j
        
        # 判断是否为TP
        if max_idx != -1 and max_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[max_idx] = True
        else:
            fp[i] = 1
    
    # 计算累积TP和FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    # 计算精度和召回率
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)
    recall = cum_tp / len(class_gt)
    # print("类别：",class_idx,"target数量：",len(class_gt),"detect数量：",len(class_dets))
    # print("累计TP：",cum_tp[-1] if len(cum_tp) > 0 else 0,"累计FP：",cum_fp[-1] if len(cum_fp) > 0 else 0)
    
    # 计算AP（改用torchmetrics方式，采用11点插值法）
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11
    
    return ap, precision, recall


def calculate_map(all_detections, all_ground_truths, iou_thresholds=None):
    """
    计算多个IoU阈值下的mAP和每个类别的AP
    
    参数:
        all_detections: 所有检测结果的列表
        all_ground_truths: 所有真实标注的列表
        iou_thresholds: IoU阈值列表，如果为None则使用[0.5]
        
    返回:
        result_dict: 包含各种指标的字典
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    # 按类别和IoU阈值初始化结果字典
    result_dict = {
        'per_class': {},
        'overall': {}
    }
    
    # 初始化各IoU阈值下的AP列表
    for threshold in iou_thresholds:
        result_dict['overall'][f'mAP@{threshold}'] = 0.0
    
    # 初始化COCO风格的mAP (0.5:0.05:0.95)
    coco_thresholds = np.arange(0.5, 1.0, 0.05)
    result_dict['overall']['mAP@0.5:0.95'] = 0.0
    
    # 记录各类别在COCO阈值下的AP总和
    coco_ap_sum = np.zeros(CLASS_NUM)
        
    # 记录各类别不同阈值下的指标
    for c in range(CLASS_NUM):
        class_name = VOC_CLASSES[c]
        result_dict['per_class'][class_name] = {}
        
        # 计算COCO风格的AP (0.5:0.05:0.95)
        coco_ap_list = []
        for threshold in coco_thresholds:
            ap, precision, recall = calculate_ap_per_class(
                all_detections, all_ground_truths, c, threshold
            )
            coco_ap_list.append(ap)
            
        # 保存COCO风格的AP
        avg_ap = np.mean(coco_ap_list) if len(coco_ap_list) > 0 else 0.0
        result_dict['per_class'][class_name]['AP@0.5:0.95'] = avg_ap
        coco_ap_sum[c] = avg_ap
        
        # 各IoU阈值下的AP
        for threshold in iou_thresholds:
            ap, precision, recall = calculate_ap_per_class(
                all_detections, all_ground_truths, c, threshold
            )
            
            result_dict['per_class'][class_name][f'AP@{threshold}'] = ap
            
            # 计算F1分数
            if len(precision) > 0 and len(recall) > 0:
                # 找到最大F1对应的位置
                f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
                max_f1_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
                
                if len(f1_scores) > 0:
                    result_dict['per_class'][class_name][f'F1@{threshold}'] = f1_scores[max_f1_idx]
                    result_dict['per_class'][class_name][f'Precision@{threshold}'] = precision[max_f1_idx]
                    result_dict['per_class'][class_name][f'Recall@{threshold}'] = recall[max_f1_idx]
                else:
                    result_dict['per_class'][class_name][f'F1@{threshold}'] = 0.0
                    result_dict['per_class'][class_name][f'Precision@{threshold}'] = 0.0
                    result_dict['per_class'][class_name][f'Recall@{threshold}'] = 0.0
            else:
                result_dict['per_class'][class_name][f'F1@{threshold}'] = 0.0
                result_dict['per_class'][class_name][f'Precision@{threshold}'] = 0.0
                result_dict['per_class'][class_name][f'Recall@{threshold}'] = 0.0
                
            # 累加到总AP
            result_dict['overall'][f'mAP@{threshold}'] += ap
    
    # 计算mAP（各类别AP的平均值）
    for threshold in iou_thresholds:
        result_dict['overall'][f'mAP@{threshold}'] /= CLASS_NUM
    
    # 计算COCO风格的mAP
    result_dict['overall']['mAP@0.5:0.95'] = np.mean(coco_ap_sum)
        
    # 计算总体的精确率、召回率和F1分数
    result_dict['overall']['Precision'] = np.mean([
        result_dict['per_class'][VOC_CLASSES[c]]['Precision@0.5'] 
        for c in range(CLASS_NUM)
    ])
    
    result_dict['overall']['Recall'] = np.mean([
        result_dict['per_class'][VOC_CLASSES[c]]['Recall@0.5'] 
        for c in range(CLASS_NUM)
    ])
    
    result_dict['overall']['F1'] = np.mean([
        result_dict['per_class'][VOC_CLASSES[c]]['F1@0.5'] 
        for c in range(CLASS_NUM)
    ])
    
    return result_dict


class DetectionProcessor:
    """处理检测结果的类，用于批量处理图像"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # 初始化检测器
        self.predictor = Pred(
            model=model,
            score_confident=config['score_threshold'],
            p_confident=config['p_threshold'],
            nms_confident=config['nms_threshold'],
            iou_con=config['iou_threshold'],
            aspect_ratio_threshold=config.get('aspect_ratio_threshold', 3.5),
            overlap_threshold=config.get('overlap_threshold', 1.0),
            center_dist_threshold=config.get('center_dist_threshold', 0.3),
            area_ratio_threshold=config.get('area_ratio_threshold', 0.5),
            output_dir="eval_results/temp/"
        )
    
    def process_batch(self, batch_images):
        """
        处理一批图像
        
        参数:
            batch_images: 批量图像张量 [B, C, H, W]
            
        返回:
            all_detections: 所有检测结果列表
        """
        batch_detections = []
        
        with torch.no_grad():
            # 批量前向传播
            results = self.model(batch_images)
            
            # 处理每张图像的结果
            for b in range(batch_images.size(0)):
                img_results = []
                
                # 处理四个分支的结果
                for branch in range(4):
                    # 提取单个图像的分支结果
                    branch_result = results[branch][b].permute(1, 2, 0)
                    # 解码处理
                    bbox = self.predictor.Decode(branch, branch_result.clone() if branch in [1, 2] else branch_result)
                    if bbox.shape[0] > 0:
                        img_results.append(bbox)
                
                # 合并四个分支的结果
                if len(img_results) > 0:
                    bbox_results = torch.cat(img_results, dim=0)
                    
                    # 应用NMS
                    final_detections = self.predictor.NMS(bbox_results)
                    
                    # 将结果转换为列表格式 [xmin, ymin, xmax, ymax, score, class_id]
                    img_detections = []
                    for det in final_detections:
                        det_list = [det[0].item(), det[1].item(), det[2].item(), det[3].item(), det[4].item(), det[5].item()]
                        img_detections.append(det_list)
                    
                    batch_detections.extend(img_detections)
        
        return batch_detections


def evaluate_model(model, val_loader, device, config=None, detector=None, fast_eval=False):
    """
    评估模型性能，计算各种指标
    
    参数:
        model: 要评估的模型
        val_loader: 验证数据加载器
        device: 设备（'cuda'或'cpu'）
        config: 配置参数字典
        detector: 可选的检测处理器实例，如果提供则不会重新创建
        fast_eval: 是否启用快速评估模式（减少打印和文件保存操作）
    
    返回:
        metrics: 评估指标字典
    """
    if config is None:
        config = {
            'p_threshold': 0.1,              # 点存在性阈值
            'score_threshold': 0.1,          # 得分阈值
            'nms_threshold': 0.1,            # NMS阈值
            'iou_threshold': 0.35,            # IoU阈值
            'aspect_ratio_threshold': 3.5,   # 长宽比例阈值，超过此值的框会被抑制
            'overlap_threshold': 1.5,        # 重叠框判定阈值，两框重叠超过此比例的较小框面积时会被抑制
            'center_dist_threshold': 0,    # 中心点距离阈值，归一化距离小于此值时触发抑制
            'area_ratio_threshold': 0.5,     # 面积比例阈值，小于此值时小框可能被抑制
            'iou_thresholds': [0.5, 0.75],   # 计算mAP时使用的IoU阈值
            'batch_size': 16                  # 批处理大小
        }
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建或复用检测处理器
    if detector is None:
        detector = DetectionProcessor(model, config)
    
    # 收集所有预测结果和真实标注
    all_detections = []
    
    # 记录推理时间
    total_time = 0
    total_images = 0
    
    # 使用数据集中预加载的标注数据，如果有的话
    if hasattr(val_loader.dataset, 'ground_truth_boxes') and val_loader.dataset.ground_truth_boxes is not None:
        all_ground_truths = val_loader.dataset.ground_truth_boxes
        if not fast_eval:
            print(f"使用预加载的 {len(all_ground_truths)} 个标注框")
    else:
        # 回退到从文件读取标注
        test_file = val_loader.dataset.test_file if hasattr(val_loader.dataset, 'test_file') else 'voctest4481.txt'
        if not fast_eval:
            print(f"从{test_file}中读取标注...")
        with open(test_file, 'r') as f:
            lines = f.readlines()
        all_ground_truths = extract_boxes_from_targets(test_file, 0, len(lines))
        if not fast_eval:
            print(f"读取了{len(all_ground_truths)}个标注框")
    
    if not fast_eval:
        print("开始批量评估模型性能...")
    
    # 按批处理图像
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, desc="批处理推理", disable=fast_eval)):
            batch_size = images.size(0)
            total_images += batch_size
            
            # 记录推理开始时间
            start_time = time.time()
            
            # 将图像移至GPU
            images = images.to(device)
            
            # 批量处理图像
            batch_detections = detector.process_batch(images)
            all_detections.extend(batch_detections)
            
            # 记录推理结束时间
            inference_time = time.time() - start_time
            total_time += inference_time
    
    # 计算平均推理时间
    avg_time_per_image = total_time / max(total_images, 1)
    fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0
    
    if not fast_eval:
        print(f"平均每张图像推理时间: {avg_time_per_image * 1000:.2f}ms (FPS: {fps:.2f})")
        print(f"检测到的边界框数量: {len(all_detections)}")
        print(f"目标边界框数量: {len(all_ground_truths)}")
    
    # 计算mAP和各类别AP
    metrics = calculate_map(all_detections, all_ground_truths, config['iou_thresholds'])
    
    # 添加推理速度指标
    metrics['overall']['inference_time_ms'] = avg_time_per_image * 1000
    metrics['overall']['fps'] = fps
    
    # 在快速评估模式下，不保存检测结果
    if not fast_eval:
        # 保存检测和真实标注，供PR曲线绘制使用
        metrics['all_detections'] = all_detections
        metrics['all_ground_truths'] = all_ground_truths
    
    return metrics


def plot_precision_recall_curve(all_detections, all_ground_truths, class_idx=None, save_path=None):
    """
    绘制精确率-召回率曲线
    
    参数:
        all_detections: 所有检测结果
        all_ground_truths: 所有真实标注
        class_idx: 类别索引，如果为None则绘制所有类别
        save_path: 保存路径，如果为None则显示图像
    """
    plt.figure(figsize=(10, 8))
    
    # 如果指定了类别，只绘制该类别的曲线
    if class_idx is not None:
        class_indices = [class_idx]
    else:
        class_indices = range(CLASS_NUM)
    
    for c in class_indices:
        ap, precision, recall = calculate_ap_per_class(
            all_detections, all_ground_truths, c, 0.5
        )
        
        if len(precision) > 0 and len(recall) > 0:
            plt.plot(recall, precision, label=f'{VOC_CLASSES[c]} (AP={ap:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"已保存精确率-召回率曲线至 {save_path}")
    else:
        plt.show()


def main(args=None):
    """
    主函数
    
    参数:
        args: 参数字典，如果为None则使用默认值
    """
    # 验证函数参数配置
    if args is None:
        args = {
            'model_path': r"/root/autodl-tmp/PLN1/results2/pln_latest.pth",  # 模型权重路径
            'test_file': 'voc_test_448.txt',        # 测试集文件
            'img_root': r"/root/autodl-tmp/JPEGImages0712/",  # 图像根目录
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 设备
            'plot_pr_curve': True,                   # 是否绘制PR曲线
            'save_results': True,                    # 是否保存结果
            'batch_size': 32,                         # 批处理大小
            'num_workers': 16,                        # 数据加载线程数
            'config': {                              # 评估配置
                'p_threshold': 0.1,                  # 点存在性阈值
                'score_threshold': 0.1,              # 得分阈值
                'nms_threshold': 0.1,                # NMS阈值  小于舍去
                'iou_threshold': 0.5,               # IoU阈值  大于则在NMS中舍去
                'aspect_ratio_threshold': 100,       # 长宽比例阈值，超过此值的框会被抑制
                'overlap_threshold': 1.5,            # 重叠框判定阈值，两框重叠超过此比例的较小框面积时会被抑制
                'center_dist_threshold': 0,        # 中心点距离阈值，归一化距离小于此值时触发抑制
                'area_ratio_threshold': 0.5,         # 面积比例阈值，小于此值时小框可能被抑制
                'iou_thresholds': [0.5, 0.75]        # 用于计算mAP的IoU阈值
            }
        }
    
    # 合并批处理大小到配置中
    args['config']['batch_size'] = args['batch_size']
    
    # 打印验证设置
    print("\n验证设置:")
    print(f"模型路径: {args['model_path']}")
    print(f"测试集文件: {args['test_file']}")
    print(f"图像根目录: {args['img_root']}")
    print(f"设备: {args['device']}")
    print(f"批处理大小: {args['batch_size']}")
    print(f"数据加载线程数: {args['num_workers']}")
    print("\n检测器配置:")
    print(f"点存在性阈值 (p_threshold): {args['config']['p_threshold']}")
    print(f"得分阈值 (score_threshold): {args['config']['score_threshold']}")
    print(f"NMS阈值 (nms_threshold): {args['config']['nms_threshold']}")
    print(f"IoU阈值 (iou_threshold): {args['config']['iou_threshold']}")
    print(f"长宽比例阈值 (aspect_ratio_threshold): {args['config']['aspect_ratio_threshold']}")
    print(f"重叠框判定阈值 (overlap_threshold): {args['config']['overlap_threshold']}")
    print(f"中心点距离阈值 (center_dist_threshold): {args['config']['center_dist_threshold']}")
    print(f"面积比例阈值 (area_ratio_threshold): {args['config']['area_ratio_threshold']}")
    
    # 加载模型
    print("\n正在加载模型...")
    model = pretrained_inception().to(args['device'])
    
    # 加载模型权重
    checkpoint = torch.load(args['model_path'], map_location=args['device'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"成功加载模型，轮次: {checkpoint['epoch']}")
    
    # 读取测试文件，获取图像路径
    image_paths = []
    with open(args['test_file'], 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                img_path = os.path.join(args['img_root'], parts[0])
                image_paths.append(img_path)
    
    print(f"从{args['test_file']}中读取了{len(image_paths)}张图像")
    
    # 创建数据集和数据加载器
    eval_dataset = EvalDataset(
        image_paths, 
        test_file=args['test_file']
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers'],
        pin_memory=True
    )
    
    # 确保评估结果目录存在
    os.makedirs("eval_results/temp", exist_ok=True)
    
    # 评估模型性能
    start_time = time.time()
    metrics = evaluate_model(model, eval_loader, args['device'], args['config'])
    eval_time = time.time() - start_time
    
    # 打印总评估时间
    print(f"\n评估完成，总耗时: {eval_time:.2f}秒")
    
    # 打印mAP和总体指标
    print("\n整体性能:")
    for metric, value in metrics['overall'].items():
        if isinstance(value, (int, float)):
            if metric == 'mAP@0.5:0.95':
                print(f"\n{metric} (COCO风格的mAP): {value:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
    
    # 打印各类别的AP
    print("\n各类别AP@0.5:")
    for c in range(CLASS_NUM):
        class_name = VOC_CLASSES[c]
        ap = metrics['per_class'][class_name]['AP@0.5']
        print(f"{class_name}: {ap:.4f}")
    
    # 打印各类别的COCO风格AP
    print("\n各类别AP@0.5:0.95 (COCO风格):")
    for c in range(CLASS_NUM):
        class_name = VOC_CLASSES[c]
        ap = metrics['per_class'][class_name]['AP@0.5:0.95']
        print(f"{class_name}: {ap:.4f}")
    
    # 保存结果
    if args['save_results']:
        # 创建结果目录
        os.makedirs('eval_results', exist_ok=True)
        
        # 生成时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        # 保存评估指标
        result_path = f"eval_results/metrics_{timestamp}.json"
        
        # 将numpy数组转换为列表以便JSON序列化
        json_metrics = {
            'per_class': {},
            'overall': {}
        }
        
        # 处理每个类别的指标
        for class_name in metrics['per_class']:
            json_metrics['per_class'][class_name] = {}
            for metric in metrics['per_class'][class_name]:
                if isinstance(metrics['per_class'][class_name][metric], np.float64):
                    json_metrics['per_class'][class_name][metric] = float(metrics['per_class'][class_name][metric])
                else:
                    json_metrics['per_class'][class_name][metric] = metrics['per_class'][class_name][metric]
        
        # 处理总体指标
        for metric in metrics['overall']:
            if isinstance(metrics['overall'][metric], np.float64):
                json_metrics['overall'][metric] = float(metrics['overall'][metric])
            elif isinstance(metrics['overall'][metric], (int, float, str, bool)):
                json_metrics['overall'][metric] = metrics['overall'][metric]
        
        # 保存配置信息
        json_metrics['config'] = args['config']
        
        with open(result_path, 'w') as f:
            json.dump(json_metrics, f, indent=4)
        
        print(f"\n评估指标已保存至 {result_path}")
        
        # 绘制PR曲线
        if args['plot_pr_curve']:
            # 直接使用已经计算得到的检测结果和真实标注
            pr_curve_path = f"eval_results/pr_curve_{timestamp}.png"
            plot_precision_recall_curve(metrics['all_detections'], 
                                       metrics['all_ground_truths'],
                                       save_path=pr_curve_path)
    
    return metrics


if __name__ == "__main__":
    main() 