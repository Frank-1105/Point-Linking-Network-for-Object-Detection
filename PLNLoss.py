import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# 忽略警告消息
warnings.filterwarnings('ignore')

# 类别数量（使用自己的数据集时需要更改）
CLASS_NUM = 20


class PLNLoss(nn.Module):
    """
    点链接网络(Point Linking Network)的损失函数
    
    计算PLN网络预测结果与标签之间的损失，包括：
    1. 点存在性损失(p_loss)：检测点是否存在的损失
    2. 坐标损失(coord_loss)：点坐标位置的预测损失
    3. 链接损失(link_loss)：点之间链接关系的预测损失
    4. 类别损失(class_loss)：物体类别预测的损失
    5. 负样本损失(noobj_loss)：没有目标的位置的预测损失
    
    注意：预测张量和目标张量形状不同
    - pred_tensor: 形状为(batch_size, 204, 14, 14)
    - target_tensor: 形状为(batch_size, 14, 14, 204)
    
    其中204维包含4个点的信息：
    - 每个点51维特征 (1维置信度 + 2维坐标 + 28维链接 + 20维类别)
    - 包含2个中心点和2个角点，每个点预测一个边界框
    - 中心点：物体中心位置
    - 角点：边界框的对角点位置
    """
    
    def __init__(self, S, B, w_class, w_coord, w_link):
        """
        初始化PLN损失函数
        
        参数:
            S (int): 网格大小，默认为14
            B (int): 每个网格预测的边界框数量，默认为2
            w_class (float): 类别损失权重
            w_coord (float): 坐标损失权重
            w_link (float): 链接损失权重
        """
        super(PLNLoss, self).__init__()
        self.S = S              # 网格大小，默认为14
        self.B = B              # 边界框数量，默认为2
        self.w_class = w_class  # 类别损失权重
        self.w_coord = w_coord  # 坐标损失权重
        self.w_link = w_link    # 链接损失权重
        self.classes = CLASS_NUM  # 类别数量
        self.noobj_scale = 0.04   # 负样本损失缩放因子
        
        # 特征维度
        self.feature_size = 51  # 每个点的特征维度 (1+2+28+20)
        self.num_points = 4     # 点的数量（2个中心点和2个角点）

    def forward(self, pred_tensor, target_tensor):
        """
        前向传播，计算损失
        
        参数:
            pred_tensor (Tensor): 预测张量，形状为(batch_size, 204, 14, 14)
            target_tensor (Tensor): 目标张量，形状为(batch_size, 14, 14, 204)
            
        返回:
            Tensor: 总损失值
            Dict: 包含各个子损失的字典 {'p_loss', 'coord_loss', 'link_loss', 'class_loss', 'noobj_loss'}
        """
        # 设备信息
        device = pred_tensor.device
        batch_size = pred_tensor.size(0)
        
        # 调整预测张量的形状以匹配目标张量的布局
        # 从(batch_size, 204, 14, 14)变为(batch_size, 14, 14, 204)
        pred_tensor_adjusted = pred_tensor.permute(0, 2, 3, 1)  # 调整维度顺序
        
        # 将张量拆分为4个点的特征
        p_loss = 0
        coord_loss = 0
        link_loss = 0
        class_loss = 0
        noobj_loss = 0
        
        # 分别处理4个点的特征
        for i in range(4):
            # 提取当前点的特征
            start_idx = i * self.feature_size
            end_idx = (i + 1) * self.feature_size
            
            # 提取当前点的所有特征 (batch_size, 14, 14, 51)
            pred_point = pred_tensor_adjusted[:, :, :, start_idx:end_idx]
            target_point = target_tensor[:, :, :, start_idx:end_idx]
            
            # 获取置信度通道（第一个通道）(batch_size, 14, 14)
            pred_conf = pred_point[:, :, :, 0]
            target_conf = target_point[:, :, :, 0]
            
            # 提取负样本掩码 (batch_size, 14, 14)
            neg_mask = target_conf == 0
            
            # 计算无目标损失
            if neg_mask.any():
                # 只对负样本计算置信度损失 (batch_size, 14, 14)
                squared_error = (pred_conf - target_conf) ** 2
                masked_error = squared_error * neg_mask.float()
                noobj_loss += masked_error.sum()
            
            # 提取正样本掩码 (batch_size, 14, 14)
            pos_mask = ~neg_mask
            
            if pos_mask.any():
                # 提取有目标的置信度
                pred_conf_pos = pred_conf[pos_mask]   # (n_positive,)
                target_conf_pos = target_conf[pos_mask]   # (n_positive,)
                
                # 计算点存在性损失
                p_loss += F.mse_loss(pred_conf_pos, target_conf_pos, reduction='sum')
                
                # 提取有目标的坐标 (2个坐标通道)
                for j in range(2):
                    pred_coord = pred_point[:, :, :, j+1]   # (batch_size, 14, 14)
                    target_coord = target_point[:, :, :, j+1]   # (batch_size, 14, 14)
                    pred_coord_pos = pred_coord[pos_mask]   # (n_positive,)
                    target_coord_pos = target_coord[pos_mask]   # (n_positive,)
                    coord_loss += F.mse_loss(pred_coord_pos, target_coord_pos, reduction='sum')
                
                # 提取有目标的链接特征 (28个链接通道)
                for j in range(28):
                    pred_link = pred_point[:, :, :, j+3]   # (batch_size, 14, 14)
                    target_link = target_point[:, :, :, j+3]   # (batch_size, 14, 14)
                    pred_link_pos = pred_link[pos_mask]   # (n_positive,)
                    target_link_pos = target_link[pos_mask]   # (n_positive,)
                    link_loss += F.mse_loss(pred_link_pos, target_link_pos, reduction='sum')
                
                # 提取有目标的类别特征 (20个类别通道)
                for j in range(20):
                    pred_class = pred_point[:, :, :, j+31]   # (batch_size, 14, 14)
                    target_class = target_point[:, :, :, j+31]   # (batch_size, 14, 14)
                    pred_class_pos = pred_class[pos_mask]   # (n_positive,)
                    target_class_pos = target_class[pos_mask]   # (n_positive,)
                    class_loss += F.mse_loss(pred_class_pos, target_class_pos, reduction='sum')
        
        # 如果没有检测到点，返回零损失
        if p_loss == 0 and coord_loss == 0 and link_loss == 0 and class_loss == 0:
            zero_tensor = torch.tensor(0.0, device=device)
            losses_dict = {
                'p_loss': zero_tensor,
                'coord_loss': zero_tensor,
                'link_loss': zero_tensor,
                'class_loss': zero_tensor,
                'noobj_loss': zero_tensor
            }
            return zero_tensor, losses_dict
        
        # 计算带权重的损失
        weighted_coord_loss = self.w_coord * coord_loss
        weighted_class_loss = self.w_class * class_loss
        weighted_link_loss = self.w_link * link_loss
        weighted_noobj_loss = self.noobj_scale * noobj_loss
        
        # 组合总损失
        total_loss = (
            p_loss + 
            weighted_coord_loss + 
            weighted_class_loss + 
            weighted_link_loss + 
            weighted_noobj_loss
        )
        
        # 返回总损失和各个子损失字典
        losses_dict = {
            'p_loss': p_loss,
            'coord_loss': coord_loss,
            'link_loss': link_loss,
            'class_loss': class_loss,
            'noobj_loss': noobj_loss,
            'weighted_coord_loss': weighted_coord_loss,
            'weighted_class_loss': weighted_class_loss,
            'weighted_link_loss': weighted_link_loss,
            'weighted_noobj_loss': weighted_noobj_loss
        }
        
        return total_loss, losses_dict


if __name__ == '__main__':
    # 测试配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    S = 14
    B = 2
    feature_dim = 204  # 51*4

    # 1. 创建模拟预测和目标张量
    pred_tensor = torch.randn(batch_size, feature_dim, S, S, device=device)  # 随机预测值
    target_tensor = torch.zeros(batch_size, S, S, feature_dim, device=device)  # 全零目标

    # 2. 随机生成正样本位置（约10%的网格有目标）
    pos_mask = torch.rand(batch_size, S, S) < 0.1

    # 3. 为有目标的位置填充合理的目标值
    for b in range(batch_size):
        for i in range(S):
            for j in range(S):
                if pos_mask[b, i, j]:
                    # 设置存在性标志为1（每个点的第一个特征）
                    for k in range(4):  # 4个点
                        target_tensor[b, i, j, k * 51] = 1
                    # 坐标 (0~1范围)
                    target_tensor[b, i, j, 1:3] = torch.rand(2)
                    # 链接 (模拟概率)
                    target_tensor[b, i, j, 3:31] = torch.rand(28)
                    # 类别 (one-hot)
                    rand_class = torch.randint(0, 20, (1,))
                    target_tensor[b, i, j, 31:51] = F.one_hot(rand_class, 20).float()

    # 4. 初始化损失函数
    criterion = PLNLoss(S=S, B=B, w_class=0, w_coord=0, w_link=1)
    criterion.to(device)

    # 5. 计算损失
    loss, losses_dict = criterion(pred_tensor, target_tensor)

    # 6. 打印验证结果
    print(f"预测张量形状: {pred_tensor.shape}")
    print(f"目标张量形状: {target_tensor.shape}")
    print(
        f"正样本数量: {pos_mask.sum().item()}/{batch_size * S * S} (约{pos_mask.sum().item() / (batch_size * S * S) * 100:.1f}%)")
    print(f"总损失值: {loss.item():.4f}")
    for loss_name, loss_value in losses_dict.items():
        print(f"{loss_name}: {loss_value.item():.4f}")

    # # 7. 反向传播测试
    # pred_tensor.requires_grad_(True)
    # loss.backward()
    # print(f"梯度检查: pred_tensor.grad norm = {pred_tensor.grad.norm().item():.4f}")

    # 8. 极端情况测试
    print("\n极端情况测试:")
    # 情况1: 所有网格都是负样本
    empty_target = torch.zeros_like(target_tensor)
    empty_loss, empty_losses_dict = criterion(pred_tensor, empty_target)
    print(f"全负样本损失: {empty_loss.item():.4f} (应接近0)")
    for loss_name, loss_value in empty_losses_dict.items():
        print(f"{loss_name}: {loss_value.item():.4f}")

    # 情况2: 预测与目标完全一致
    perfect_pred = target_tensor.permute(0, 3, 1, 2).clone()
    perfect_loss, perfect_losses_dict = criterion(perfect_pred, target_tensor)
    print(f"完美预测损失: {perfect_loss.item():.4f} (应为0)")
    for loss_name, loss_value in perfect_losses_dict.items():
        print(f"{loss_name}: {loss_value.item():.4f}")
