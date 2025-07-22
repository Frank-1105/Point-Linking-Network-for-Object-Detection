import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime
from tqdm import tqdm
import numpy as np

from PLNdata import PLNDataset  # 使用优化后的类名
from PLNLoss import PLNLoss  # 使用优化后的类名
from PLNnet import pretrained_inception
from val import evaluate_model, EvalDataset


class Logger(object):
    """日志记录器：同时输出到控制台和文件，但可以控制哪些信息仅显示在控制台"""
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
        self.log_to_file = True  # 控制是否同时写入文件

    def write(self, message):
        self.terminal.write(message)
        if self.log_to_file:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def set_log_to_file(self, log_to_file):
        """设置是否同时写入文件"""
        self.log_to_file = log_to_file


def setup_logger():
    """设置日志系统"""
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = Logger(log_filename)
    sys.stdout = logger
    print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # 创建一个额外的JSON格式日志文件用于记录训练指标
    metrics_log_filename = f"metrics_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return logger, log_filename, metrics_log_filename


def save_checkpoint(model, optimizer, epoch, loss, is_best=False, other_info=None):
    """
    保存模型检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        epoch: 当前训练轮次
        loss: 当前验证损失
        is_best: 是否为最佳模型
        other_info: 其他需要保存的信息
    """
    try:
        model.eval()
        # 保存路径
        checkpoint_dir = "results2"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        
        latest_ckpt_path = os.path.join(checkpoint_dir, "pln_latest.pth")
        best_ckpt_path = os.path.join(checkpoint_dir, "pln_best.pth")
        
        # 准备保存内容
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        # 添加其他信息
        if other_info:
            checkpoint.update(other_info)
        
        # 保存最新检查点
        torch.save(checkpoint, latest_ckpt_path)
        print(f"已保存最新检查点到 {latest_ckpt_path}")
        
        # 如果是最佳模型，另存一份
        if is_best:
            torch.save(checkpoint, best_ckpt_path)
            print(f"已保存最佳模型到 {best_ckpt_path}")
            
    except Exception as e:
        print(f"保存检查点失败: {str(e)}")
    finally:
        model.train()


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    加载模型检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        device: 设备
        
    返回:
        start_epoch: 开始轮次
        best_loss: 最佳损失
        其他恢复的信息
    """
    print(f"正在加载检查点 {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        
        # 恢复其他信息
        current_lr = checkpoint.get('current_lr', None)
        total_iterations = checkpoint.get('total_iterations', 0)
        
        print(f"成功加载检查点，从epoch {start_epoch}继续训练")
        if current_lr:
            print(f"当前学习率: {current_lr:.6f}")
        print(f"总迭代次数: {total_iterations}")
        
        return start_epoch, best_loss, {'current_lr': current_lr, 'total_iterations': total_iterations}
    except Exception as e:
        print(f"加载检查点失败: {str(e)}")
        return 0, float('inf'), {'current_lr': None, 'total_iterations': 0}


def adjust_learning_rate(optimizer, current_iteration, init_lr=0.001, max_lr=0.005, 
                         warmup_iterations=50000, min_lr=0.0004, total_decay_iterations=100000):
    """
    学习率调整策略:
    1. 预热阶段：从init_lr线性增加到max_lr
    2. 衰减阶段：从max_lr按余弦退火衰减到min_lr
    
    参数:
        optimizer: 优化器
        current_iteration: 当前迭代次数
        init_lr: 初始学习率
        max_lr: 最大学习率（预热目标）
        warmup_iterations: 预热迭代次数
        min_lr: 衰减阶段的最小学习率
        total_decay_iterations: 衰减阶段的总迭代次数（从warmup结束后开始计算）
    
    返回:
        当前学习率
    """
    if current_iteration < warmup_iterations:
        # 1. 预热阶段：线性增加
        lr = init_lr + (max_lr - init_lr) * (current_iteration / warmup_iterations)
    else:
        # 2. 衰减阶段：余弦退火
        decay_iteration = current_iteration - warmup_iterations
        progress = min(decay_iteration / total_decay_iterations, 1.0)  # 限制在[0,1]
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    
    # 更新优化器的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def log_metrics(metrics_file, epoch, train_loss, val_metrics=None, component_losses=None, lr=None):
    """
    记录训练和验证指标
    
    参数:
        metrics_file: 指标日志文件路径
        epoch: 当前轮次
        train_loss: 训练损失
        val_metrics: 验证指标字典，包含val_loss和其他评估指标
        component_losses: 组件损失字典
        lr: 当前学习率
    """
    # 读取现有指标记录
    metrics = []
    try:
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
    except Exception:
        metrics = []
    
    # 准备当前轮次的指标记录
    epoch_metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if lr is not None:
        epoch_metrics['learning_rate'] = lr
    
    # 添加组件损失
    if component_losses:
        for key, value in component_losses.items():
            epoch_metrics[f'train_{key}'] = value
    
    # 添加验证指标
    if val_metrics:
        # 添加验证损失
        if 'val_loss' in val_metrics:
            epoch_metrics['val_loss'] = val_metrics['val_loss']
            
        # 添加评估指标
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                epoch_metrics[key] = value
    
    # 添加到记录列表
    metrics.append(epoch_metrics)
    
    # 保存指标记录
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                total_iterations, num_epochs, logger, init_lr=0.001, max_lr=0.005, log_interval=5):
    """
    训练一个epoch
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        total_iterations: 总迭代次数
        num_epochs: 总训练轮数
        logger: 日志记录器对象
        init_lr: 初始学习率
        max_lr: 最大学习率
        log_interval: 日志记录间隔（迭代次数）
        
    返回:
        avg_loss: 平均损失
        updated_total_iterations: 更新后的总迭代次数
        current_lr: 当前学习率
        batch_losses: 每个批次的损失
        avg_component_losses: 平均组件损失字典
    """
    model.train()
    total_loss = 0
    batch_losses = []  # 用于记录每个批次的损失
    
    # 初始化各个组件损失的累加器
    component_losses = {
        'p_loss': 0, 
        'coord_loss': 0,
        'link_loss': 0,
        'class_loss': 0,
        'noobj_loss': 0,
        'weighted_coord_loss': 0,
        'weighted_class_loss': 0,
        'weighted_link_loss': 0,
        'weighted_noobj_loss': 0
    }
    
    # 设置进度条，禁用tqdm从而手动控制进度条更新
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=True, dynamic_ncols=True)
    current_lr = None
    
    # 修改日志行为，进度条显示时不记录到文件
    logger.set_log_to_file(False)
    
    for i, (images, target) in enumerate(pbar):
        # 更新总迭代次数
        total_iterations += 1
        
        # 调整学习率
        current_lr = adjust_learning_rate(optimizer, total_iterations, init_lr, max_lr)
        
        # 每100次迭代打印一次学习率（仅控制台显示，不记录到日志）
        if total_iterations % 100 == 0:
            print(f'迭代次数: {total_iterations}, 当前学习率: {current_lr:.6f}')

        # 前向传播和损失计算
        images, target = images.to(device), target.to(device)
        pred = model(images)    # pred:[4,batch_size,204,14,14]
        target = target.permute(1, 0, 2, 3, 4)  # [batch_size,4,14,14,204]-->[4,batch_size,14,14,204]

        batch_size = pred[0].shape[0]
        
        # 计算四个点的损失
        loss0, losses_dict0 = criterion(pred[0], target[0])
        loss1, losses_dict1 = criterion(pred[1], target[1])
        loss2, losses_dict2 = criterion(pred[2], target[2])
        loss3, losses_dict3 = criterion(pred[3], target[3])
        
        # 合并四个点的损失字典
        for key in component_losses.keys():
            if key in losses_dict0:
                # 累加每个点的损失并平均
                point_loss = (losses_dict0[key] + losses_dict1[key] + 
                              losses_dict2[key] + losses_dict3[key]) / batch_size
                component_losses[key] += point_loss.item()
        
        loss = (loss0 + loss1 + loss2 + loss3) / batch_size
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        loss_value = loss.item()
        total_loss += loss_value
        batch_losses.append(loss_value)
        avg_loss = total_loss / (i + 1)
        
        # 计算平均组件损失
        avg_component_losses = {k: v / (i + 1) for k, v in component_losses.items()}
        
        # 动态更新进度条，不写入日志文件
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
        # 定期记录训练信息到终端，但不写入日志文件
        if (i + 1) % log_interval == 0 and i > 0:
            # 只在控制台显示，不写入日志
            print(f'\r迭代 [{i+1}/{len(train_loader)}], Loss: {loss_value:.4f}, Avg: {avg_loss:.4f}, LR: {current_lr:.6f}', end='')
    
    # 恢复日志行为，之后的输出同时记录到文件
    logger.set_log_to_file(True)
    
    # 计算并返回平均损失
    epoch_avg_loss = total_loss / len(train_loader)
    avg_component_losses = {k: v / len(train_loader) for k, v in component_losses.items()}
    
    # 在epoch结束时记录摘要信息到日志文件
    print(f'\nEpoch {epoch+1}/{num_epochs} 训练完成, 平均损失: {epoch_avg_loss:.5f}')
    print('平均组件损失:')
    for k, v in avg_component_losses.items():
        print(f'  {k}: {v:.5f}')
    
    return epoch_avg_loss, total_iterations, current_lr, batch_losses, avg_component_losses


def validate(model, val_loader, device, epoch, test_file, img_root, config):
    """
    使用val.py的评估功能进行验证
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        epoch: 当前轮次
        test_file: 测试文件路径
        img_root: 图像根目录
        config: 配置参数
        
    返回:
        val_metrics: 包含验证指标的字典
    """
    print(f"\n开始评估 Epoch {epoch+1}...")
    
    # 设置验证配置
    val_config = {
        'p_threshold': config.get('p_threshold', 0.1),
        'score_threshold': config.get('score_threshold', 0.1),
        'nms_threshold': config.get('nms_threshold', 0.1),
        'iou_threshold': config.get('iou_threshold', 0.35),
        'aspect_ratio_threshold': config.get('aspect_ratio_threshold', 3.5),
        'overlap_threshold': config.get('overlap_threshold', 1.5),
        'center_dist_threshold': config.get('center_dist_threshold', 0.3),
        'area_ratio_threshold': config.get('area_ratio_threshold', 0.5),
        'iou_thresholds': config.get('iou_thresholds', [0.5, 0.75]),
        'batch_size': config.get('batch_size', 8)
    }
    
    # 静态变量缓存检测器实例，避免重复创建
    if not hasattr(validate, 'detector'):
        # 第一次调用时初始化检测处理器
        from val import DetectionProcessor
        validate.detector = DetectionProcessor(model, val_config)
    
    # 执行评估，使用快速评估模式
    metrics = evaluate_model(
        model, 
        val_loader, 
        device, 
        val_config, 
        detector=validate.detector,
        fast_eval=True
    )
    
    # 提取验证损失和性能指标
    val_metrics = {
        'val_loss': 0.0,  # 由于evaluate_model不返回损失，可以设为0或从其他地方获取
        'mAP@0.5': metrics['overall']['mAP@0.5'],
        'mAP@0.75': metrics['overall'].get('mAP@0.75', 0.0),
        'mAP@0.5:0.95': metrics['overall'].get('mAP@0.5:0.95', 0.0),
        'Precision': metrics['overall']['Precision'],
        'Recall': metrics['overall']['Recall'],
        'F1': metrics['overall']['F1'],
        'FPS': metrics['overall']['fps']
    }
    
    # 打印主要指标
    print("\n验证结果:")
    print(f"mAP@0.5: {val_metrics['mAP@0.5']:.4f}")
    print(f"mAP@0.75: {val_metrics['mAP@0.75']:.4f}")
    if 'mAP@0.5:0.95' in val_metrics:
        print(f"mAP@0.5:0.95 (COCO风格): {val_metrics['mAP@0.5:0.95']:.4f}")
    print(f"精确率: {val_metrics['Precision']:.4f}")
    print(f"召回率: {val_metrics['Recall']:.4f}")
    print(f"F1分数: {val_metrics['F1']:.4f}")
    print(f"推理速度: {val_metrics['FPS']:.2f} FPS")
    
    # 保存评估结果
    save_path = f"results/{epoch+1}"
    os.makedirs(save_path, exist_ok=True)
    
    # 生成时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    result_path = f"{save_path}/eval_metrics_{timestamp}.json"
    
    # 将numpy值转换为Python原生类型以便JSON序列化
    json_metrics = {
        'overall': {}
    }
    
    # 只保存总体指标，减少处理时间
    for metric in metrics['overall']:
        if isinstance(metrics['overall'][metric], np.float64):
            json_metrics['overall'][metric] = float(metrics['overall'][metric])
        elif isinstance(metrics['overall'][metric], (int, float, str, bool)):
            json_metrics['overall'][metric] = metrics['overall'][metric]
    
    # 保存验证结果
    with open(result_path, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    print(f"验证指标已保存至 {result_path}")
    
    return val_metrics


def train(config):
    """
    训练主函数
    
    参数:
        config: 配置字典，包含训练参数
    """
    # 设置设备
    device = config['device']
    print(f"Using device: {device}")
    
    # 初始化日志系统
    logger, log_filename, metrics_log_filename = setup_logger()
    print(f"训练指标将被记录到: {metrics_log_filename}")
    
    # 数据集加载
    train_dataset = PLNDataset(
        img_root=config['data_root'], 
        list_file=config['train_list'],
        train=True, 
        transform=[transforms.ToTensor()]
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 创建用于验证的数据集和加载器
    # 读取测试文件，获取图像路径
    image_paths = []
    with open(config['val_list'], 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                img_path = os.path.join(config['data_root'], parts[0])
                image_paths.append(img_path)
    
    # 创建验证数据集和数据加载器
    val_dataset = EvalDataset(
        image_paths, 
        test_file=config['val_list']  # 传递测试文件路径
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['eval_batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f'训练集包含 {len(train_dataset)} 张图像')
    print(f'验证集包含 {len(val_dataset)} 张图像')
    
    # 模型初始化
    model = pretrained_inception().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['init_lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    # optimizer = optim.RMSprop(
    #     model.parameters(),
    #     lr=config['init_lr'],
    #     momentum=config['momentum'],
    #     weight_decay=config['weight_decay']
    # )
    # 损失函数
    # optimizer = optim.Adam(
    #         model.parameters(),
    #         lr=config['init_lr'],
    #         weight_decay=config['weight_decay'],
    # )
    criterion = PLNLoss(
        S=config['grid_size'],
        B=config['num_boxes'],
        w_coord=config['w_coord'],
        w_link=config['w_link'],
        w_class=config['w_class']
    ).to(device)
    
    # 加载检查点（如果存在）
    checkpoint_path = os.path.join("results", "pln_best.pth")
    if os.path.exists(checkpoint_path) and config['resume_training']:
        start_epoch, best_loss, other_info = load_checkpoint(model, optimizer, checkpoint_path, device)
        total_iterations = other_info.get('total_iterations', 0)
    else:
        start_epoch = 0
        best_loss = float('inf')
        total_iterations = 0
    
    # 记录训练配置
    print("\n训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 创建用于记录训练历史的字典
    history = {
        'train_loss': [],
        'val_metrics': [],
        'component_losses': []
    }
    
    # 训练循环
    for epoch in range(start_epoch, config['num_epochs']):
        # 训练一个epoch
        train_loss, total_iterations, current_lr, batch_losses, avg_component_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            total_iterations, config['num_epochs'], logger, config['init_lr'], config['max_lr'],
            log_interval=config.get('log_interval', 5)
        )
        
        # 每个epoch更新训练历史
        history['train_loss'].append(train_loss)
        history['component_losses'].append(avg_component_losses)
        
        # 是否需要进行验证
        do_validation = (epoch + 1) % config['eval_interval'] == 0 or epoch == config['num_epochs'] - 1
        
        val_metrics = None
        if do_validation:
            # 进行验证
            val_metrics = validate(
                model, val_loader, device, epoch, 
                config['val_list'], config['data_root'],
                config['eval_config']
            )
            
            # 更新验证历史
            history['val_metrics'].append(val_metrics)
            
            # 检查是否为最佳模型
            is_best = val_metrics['mAP@0.5'] > history.get('best_map', 0)
            if is_best:
                history['best_map'] = val_metrics['mAP@0.5']
                history['best_epoch'] = epoch + 1
                print(f'获得最佳mAP@0.5: {val_metrics["mAP@0.5"]:.5f}, Epoch: {epoch+1}')
        else:
            is_best = False
            # 仅记录训练损失和组件损失
            val_metrics = {'val_loss': 0.0}  # 占位符
        
        # 记录指标到日志文件
        log_metrics(metrics_log_filename, epoch+1, train_loss, val_metrics, avg_component_losses, current_lr)
        
        # 保存其他信息
        other_info = {
            'current_lr': current_lr,
            'total_iterations': total_iterations,
            'train_loss': train_loss,
            'component_losses': avg_component_losses,
            'history': history
        }
        
        if val_metrics:
            other_info['val_metrics'] = val_metrics
        
        # 每个epoch都保存检查点
        save_checkpoint(model, optimizer, epoch, train_loss, is_best, other_info)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()

    # 训练结束信息
    print("\n训练完成!")
    if 'best_map' in history:
        print(f"最佳mAP@0.5: {history['best_map']:.5f}, 在Epoch {history['best_epoch']}")
    print(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练日志保存在: {log_filename}")
    print(f"训练指标保存在: {metrics_log_filename}")


if __name__ == "__main__":
    # 训练配置
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_root': r'/root/autodl-tmp/JPEGImages0712/',
        'train_list': 'voc_original_trainval.txt',
        'val_list': 'voc_test_448.txt',
        'batch_size': 16,
        'eval_batch_size': 16,  # 评估时使用更大的批处理大小
        'num_workers': 4,
        'init_lr': 0.001,
        'max_lr': 0.005,
        'min_lr': 0.0004,
        'num_epochs': 125,
        'momentum': 0.9,
        'weight_decay': 0.00004,
        'resume_training': False,
        'grid_size': 14,
        'num_boxes': 2,
        'w_coord': 2.0,
        'w_link': 0.5,
        'w_class': 0.5,
        'log_interval': 100,  # 每5个批次记录一次训练日志
        'eval_interval': 1000,  # 每5个epoch进行一次验证
        'eval_config': {     # 验证配置
            'p_threshold': 0.1,
            'score_threshold': 0.1,
            'nms_threshold': 0.1,
            'iou_threshold': 0.35,
            'aspect_ratio_threshold': 3.5,
            'overlap_threshold': 1.5,
            'center_dist_threshold': 0.3,
            'area_ratio_threshold': 0.5,
            'iou_thresholds': [0.5, 0.75]
        }
    }
    
    # 开始训练
    train(config)