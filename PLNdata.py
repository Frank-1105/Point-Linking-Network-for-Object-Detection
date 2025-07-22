import os
import os.path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import PLNLoss
from PLNnet import inceptionresnetv2

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 根据txt文件制作ground truth
CLASS_NUM = 20  # 使用其他训练集需要更改


class PLNDataset(Dataset):
    """
    点链接网络(Point Linking Network)数据集类
    
    用于加载和处理目标检测数据，生成PLN模型所需的标签格式
    """
    image_size = 448

    def __init__(self, img_root, list_file, train, transform):
        """
        初始化数据集
        
        参数:
            img_root (str): 图像根目录
            list_file (str): 标注文件路径 (txt格式)
            train (bool): 是否为训练模式
            transform (list): 图像变换操作列表
        """
        # 基本参数设置
        self.root = img_root
        self.list_file = list_file
        self.train = train
        self.transform = transform
        
        # 存储数据的列表
        self.fnames = []    # 文件名
        self.boxes = []     # 边界框坐标 [x1, y1, x2, y2]
        self.labels = []    # 类别标签
        
        # 网络相关参数
        self.S = 14         # 网格大小
        self.B = 2          # 每个位置预测的边界框数量
        self.C = CLASS_NUM  # 类别数量
        
        # 图像预处理参数
        self.mean = (123, 117, 104)  # 像素均值
        
        # 读取标注文件
        # 标注格式: 图片名 xmin1 ymin1 xmax1 ymax1 类别1 xmin2 ymin2 xmax2 ymax2 类别2 ...
        with open(list_file, 'r') as file_txt:
            lines = file_txt.readlines()
            
        # 解析每一行数据
        for line in lines:
            splited = line.strip().split()
            
            # 存储图像文件名
            self.fnames.append(splited[0])
            
            # 计算图像中包含的边界框数量
            num_boxes = (len(splited) - 1) // 5
            
            # 临时存储当前图像的边界框和标签
            boxes = []
            labels = []
            
            # 提取每个边界框的坐标和类别
            for i in range(num_boxes):
                x1 = float(splited[1 + 5 * i])     # xmin
                y1 = float(splited[2 + 5 * i])     # ymin
                x2 = float(splited[3 + 5 * i])     # xmax
                y2 = float(splited[4 + 5 * i])     # ymax
                c = int(splited[5 + 5 * i])        # 类别 (0-19)
                
                boxes.append([x1, y1, x2, y2])
                labels.append(c)
            
            # 转换为Tensor并添加到数据列表
            self.boxes.append(torch.Tensor(boxes))
            self.labels.append(torch.LongTensor(labels))
            
        # 样本总数
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            tuple: (img, target)
                img (Tensor): 预处理后的图像
                target (Tensor): PLN格式的标签 [4, 14, 14, 204]
        """
        # 获取图像文件名和对应的边界框、标签
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root + fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        # 获取图像尺寸并归一化边界框坐标
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)  # 归一化到[0,1]

        # 图像预处理
        img = self.BGR2RGB(img)        # BGR转RGB
        img = self.subtract_mean(img)  # 减去均值
        img = cv2.resize(img, (self.image_size, self.image_size))  # 调整尺寸

        # 创建PLN格式的标签
        label_generator = LabelGenerator("", "", "", 0, 448, 14)
        target = []
        for branch in range(4):  # 四个分支分别对应四个角点
            t = label_generator.generate_label(branch, boxes, labels)
            target.append(t)


        # 堆叠成[4, 14, 14, 204]的张量
        target = torch.stack(target)

        # 应用图像变换
        for t in self.transform:
            img = t(img)
        # if self.transform:
        #     img = self.transform(img)

        return img, target


    def __len__(self):
        """返回数据集的样本数量"""
        return self.num_samples

    def BGR2RGB(self, img):
        """BGR格式转RGB格式"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def subtract_mean(self, img, mean=None):
        """
        图像减去均值
        
        参数:
            img: 输入图像
            mean: 均值，默认使用类初始化时设定的均值
        """
        if mean is None:
            mean = self.mean
        mean = np.array(mean, dtype=np.float32)
        return img - mean

    def get_val_data(self):
        """
        获取验证数据结构，用于验证函数
        
        返回:
            dict: 包含图像路径和标注信息的字典
        """
        # 构建完整的图像路径列表
        img_paths = [os.path.join(self.root, fname) for fname in self.fnames]
        
        # 返回验证所需的数据结构
        return {
            'img_paths': img_paths,         # 图像完整路径列表
            'annotations_path': self.list_file,  # 标注文件路径
            'boxes': self.boxes,            # 边界框列表
            'labels': self.labels,          # 标签列表
            'image_size': self.image_size   # 图像大小
        }


class LabelGenerator:
    """
    PLN标签生成器
    
    负责将边界框和类别标签转换为PLN格式的标签
    """
    
    def __init__(self, train_dir_obj, test_dir_obj, loader_type, seed, pic_width, S=14):
        """
        初始化标签生成器
        
        参数:
            train_dir_obj: 训练目录
            test_dir_obj: 测试目录
            loader_type: 加载类型
            seed: 随机种子
            pic_width: 图像宽度
            S: 网格大小，默认为14
        """
        self.loader_type = loader_type
        self.s = S                   # 网格大小
        self.classes = 20            # 类别数量
        self.B = 2                   # 每个位置的边界框数量
        self.infinite = 100000000    # 用于softmax的大数
        self.pic_width = pic_width   # 图像宽度
        
        # 设置随机种子
        torch.manual_seed(seed)

    def generate_label(self, branch, boxes, labels):
        """
        生成指定分支的PLN标签
        
        参数:
            branch (int): 分支索引 (0-3)，对应左下、左上、右下、右上四个角点
            boxes (Tensor): 归一化后的边界框坐标
            labels (Tensor): 类别标签
            
        返回:
            Tensor: 形状为[14,14,204]的标签张量
        """
        if len(boxes) == 0:
            # 如果没有边界框，返回全零张量
            return torch.zeros((self.s, self.s, (1 + 2 + 2*self.s + self.classes) * self.B * 2))
        
        # 计算类别相关张量 Q
        Q_tensor, Q_ct_tensor = self.generate_class_tensors(branch, boxes, labels)
        
        # 计算点存在性张量 P
        P_tensor, P_ct_tensor = self.generate_point_tensors(branch, boxes)
        
        # 计算连接关系张量 L
        Link_ct_list, Link_cr_list = self.generate_link_tensors(branch, boxes)
        
        # 计算相对位置张量 x
        x_tensor, x_ct_tensor = self.generate_position_tensors(branch, boxes)

        # 合并所有张量生成最终标签
        final_tensor = self.combine_tensors(
            Q_tensor, Q_ct_tensor, P_tensor, P_ct_tensor, 
            Link_ct_list, Link_cr_list, x_tensor, x_ct_tensor
        )
        
        return final_tensor

    def generate_point_tensors(self, branch, boxes):
        """
        生成点存在性张量
        
        参数:
            branch (int): 分支索引 (0-3)
            boxes (Tensor): 边界框坐标
            
        返回:
            tuple: (posi, posi_ct) 角点和中心点的存在性张量
        """
        posi = []
        posi_ct = []

        # 初始化张量
        p_tensor = torch.zeros((self.s, self.s, 1))
        p_ct_tensor = torch.zeros((self.s, self.s, 1))
        p_tensor1 = torch.zeros((self.s, self.s, 1))
        p_ct_tensor1 = torch.zeros((self.s, self.s, 1))
        
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            
            # 防止坐标值过大
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
                
            # 根据分支索引设置不同角点
            if branch == 0:  # 左下角
                p_tensor[int(ymax * self.s), int(xmin * self.s)] = 1
                p_tensor1[int(ymax * self.s), int(xmin * self.s)] = 1
            elif branch == 1:  # 左上角
                p_tensor[int(ymin * self.s), int(xmin * self.s)] = 1
                p_tensor1[int(ymin * self.s), int(xmin * self.s)] = 1
            elif branch == 2:  # 右下角
                p_tensor[int(ymax * self.s), int(xmax * self.s)] = 1
                p_tensor1[int(ymax * self.s), int(xmax * self.s)] = 1
            elif branch == 3:  # 右上角
                p_tensor[int(ymin * self.s), int(xmax * self.s)] = 1
                p_tensor1[int(ymin * self.s), int(xmax * self.s)] = 1
                
            # 设置中心点
            center_y = int((ymin + ymax) / 2 * self.s)
            center_x = int((xmin + xmax) / 2 * self.s)
            p_ct_tensor[center_y, center_x] = 1
            p_ct_tensor1[center_y, center_x] = 1

        posi.append(p_tensor)
        posi.append(p_tensor1)
        posi_ct.append(p_ct_tensor)
        posi_ct.append(p_ct_tensor1)

        return posi, posi_ct

    def generate_position_tensors(self, branch, boxes):
        """
        生成相对位置张量
        
        参数:
            branch (int): 分支索引 (0-3)
            boxes (Tensor): 边界框坐标
            
        返回:
            tuple: (pos_list, pos_ct_list) 角点和中心点的相对位置张量
        """
        pos_list = []
        pos_ct_list = []

        # 初始化张量
        pos_tensor = torch.zeros((self.s, self.s, 2))
        pos_ct_tensor = torch.zeros((self.s, self.s, 2))
        pos_tensor1 = torch.zeros((self.s, self.s, 2))
        pos_ct_tensor1 = torch.zeros((self.s, self.s, 2))
        
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            
            # 防止坐标值过大
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
                
            # 根据分支索引设置不同角点的相对位置
            if branch == 0:  # 左下角
                y_idx, x_idx = int(ymax * self.s), int(xmin * self.s)
                y_rel = ymax * self.s - y_idx
                x_rel = xmin * self.s - x_idx
                pos_tensor[y_idx, x_idx] = torch.tensor([y_rel, x_rel])
                pos_tensor1[y_idx, x_idx] = torch.tensor([y_rel, x_rel])
            elif branch == 1:  # 左上角
                y_idx, x_idx = int(ymin * self.s), int(xmin * self.s)
                y_rel = ymin * self.s - y_idx
                x_rel = xmin * self.s - x_idx
                pos_tensor[y_idx, x_idx] = torch.tensor([y_rel, x_rel])
                pos_tensor1[y_idx, x_idx] = torch.tensor([y_rel, x_rel])
            elif branch == 2:  # 右下角
                y_idx, x_idx = int(ymax * self.s), int(xmax * self.s)
                y_rel = ymax * self.s - y_idx
                x_rel = xmax * self.s - x_idx
                pos_tensor[y_idx, x_idx] = torch.tensor([y_rel, x_rel])
                pos_tensor1[y_idx, x_idx] = torch.tensor([y_rel, x_rel])
            elif branch == 3:  # 右上角
                y_idx, x_idx = int(ymin * self.s), int(xmax * self.s)
                y_rel = ymin * self.s - y_idx
                x_rel = xmax * self.s - x_idx
                pos_tensor[y_idx, x_idx] = torch.tensor([y_rel, x_rel])
                pos_tensor1[y_idx, x_idx] = torch.tensor([y_rel, x_rel])
                
            # 计算中心点相对位置
            ctx_p = (xmin * self.s + xmax * self.s) / 2
            cty_p = (ymin * self.s + ymax * self.s) / 2
            ctx = int(ctx_p)
            cty = int(cty_p)
            pos_ct_tensor[cty, ctx] = torch.tensor([ctx_p - ctx, cty_p - cty])
            pos_ct_tensor1[cty, ctx] = torch.tensor([ctx_p - ctx, cty_p - cty])

        pos_list.append(pos_tensor)
        pos_list.append(pos_tensor1)
        pos_ct_list.append(pos_ct_tensor)
        pos_ct_list.append(pos_ct_tensor1)
        
        return pos_list, pos_ct_list   #坐标是[x,y]，但在图片中索引保持[y,x]

    def generate_boxes_tensor(self, boxes):
        """
        处理边界框坐标，生成PLN格式的边界框张量
        
        参数:
            boxes (Tensor): 边界框坐标
            
        返回:
            Tensor: 形状为[N, 4, 2, 2]的边界框张量
        """
        if len(boxes) == 0:
            return torch.zeros((2, 4, 2, 2))
            
        boxes_list = []
        points = []
        
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            
            # 防止坐标值过大
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
                
            # 计算中心坐标
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            
            # 生成中心点与四个角点的连接关系
            # 中心点到左下角
            points.append(torch.tensor([
                (center_x * self.s, center_y * self.s), 
                (xmin * self.s, ymax * self.s)
            ]))
            
            # 中心点到左上角
            points.append(torch.tensor([
                (center_x * self.s, center_y * self.s), 
                (xmin * self.s, ymin * self.s)
            ]))
            
            # 中心点到右下角
            points.append(torch.tensor([
                (center_x * self.s, center_y * self.s), 
                (xmax * self.s, ymax * self.s)
            ]))
            
            # 中心点到右上角
            points.append(torch.tensor([
                (center_x * self.s, center_y * self.s), 
                (xmax * self.s, ymin * self.s)
            ]))
            
        points_tensor = torch.stack(points)
        boxes_list.append(points_tensor)
        
        stacked_tensor = torch.stack(boxes_list).squeeze(0)
        # 重塑为[N, 4, 2, 2]形状，表示N个边界框，每个有4个角点，每个点是2维坐标
        boxes_reshaped = stacked_tensor.view(int(stacked_tensor.shape[0] / 4), 4, 2, 2)
        
        return boxes_reshaped

    def generate_link_weights(self, center_pt, corner_pt):
        """
        生成中心点到角点的连接权重
        
        参数:
            center_pt (Tensor): 中心点坐标
            corner_pt (Tensor): 角点坐标
            
        返回:
            tuple: (L_ct, L_cr) 连接权重张量
        """
        # 初始化x和y方向的权重张量
        Lx_ct = torch.zeros(self.s)
        Ly_ct = torch.zeros(self.s)
        
        # 设置中心点到角点的连接权重
        Lx_ct[int(corner_pt[0])] = self.infinite
        Ly_ct[int(corner_pt[1])] = self.infinite
        
        # 连接x和y方向的权重
        L_ct = torch.cat((Lx_ct, Ly_ct))
        
        # 初始化角点到中心点的权重张量
        Lx_cr = torch.zeros(self.s)
        Ly_cr = torch.zeros(self.s)
        
        # 设置角点到中心点的连接权重
        Lx_cr[int(center_pt[0])] = self.infinite
        Ly_cr[int(center_pt[1])] = self.infinite
        
        # 连接x和y方向的权重
        L_cr = torch.cat((Lx_cr, Ly_cr))
        
        return L_ct, L_cr

    def generate_link_tensors(self, branch, boxes):
        """
        生成连接关系张量
        
        参数:
            branch (int): 分支索引 (0-3)
            boxes (Tensor): 边界框坐标
            
        返回:
            tuple: (Link_ct_list, Link_cr_list) 连接关系张量列表
        """
        Link_ct_list = []
        Link_cr_list = []
        
        # 初始化张量
        Link_tmp_cr = torch.zeros((self.s, self.s, 2 * self.s))
        Link_tmp_ct = torch.zeros((self.s, self.s, 2 * self.s))
        Link_tmp_cr1 = torch.zeros((self.s, self.s, 2 * self.s))
        Link_tmp_ct1 = torch.zeros((self.s, self.s, 2 * self.s))
        
        # 生成边界框张量   [N, 4, 2, 2]
        box_tensor = self.generate_boxes_tensor(boxes)
        
        # 提取指定分支的边界框
        box_branch = box_tensor[:, branch, ...]
        
        # 为每个边界框生成连接权重
        for obj_idx, obj_data in enumerate(box_branch):
            # 获取中心点和角点的连接权重
            L_ct, L_cr = self.generate_link_weights(obj_data[0], obj_data[1])
            
            # 在对应位置设置权重
            corner_y, corner_x = int(obj_data[1][1]), int(obj_data[1][0])
            center_y, center_x = int(obj_data[0][1]), int(obj_data[0][0])
            
            Link_tmp_cr[corner_y, corner_x] = L_cr
            Link_tmp_cr1[corner_y, corner_x] = L_cr
            Link_tmp_ct[center_y, center_x] = L_ct
            Link_tmp_ct1[center_y, center_x] = L_ct
        
        # 应用softmax激活，分别对x和y方向进行
        Link_tmp_ct[..., :self.s] = F.softmax(Link_tmp_ct[..., :self.s], dim=-1)
        Link_tmp_ct[..., self.s:] = F.softmax(Link_tmp_ct[..., self.s:], dim=-1)
        Link_tmp_ct1[..., :self.s] = F.softmax(Link_tmp_ct1[..., :self.s], dim=-1)
        Link_tmp_ct1[..., self.s:] = F.softmax(Link_tmp_ct1[..., self.s:], dim=-1)
        
        Link_tmp_cr[..., :self.s] = F.softmax(Link_tmp_cr[..., :self.s], dim=-1)
        Link_tmp_cr[..., self.s:] = F.softmax(Link_tmp_cr[..., self.s:], dim=-1)
        Link_tmp_cr1[..., :self.s] = F.softmax(Link_tmp_cr1[..., :self.s], dim=-1)
        Link_tmp_cr1[..., self.s:] = F.softmax(Link_tmp_cr1[..., self.s:], dim=-1)
        
        Link_ct_list.append(Link_tmp_ct)
        Link_ct_list.append(Link_tmp_ct1)
        Link_cr_list.append(Link_tmp_cr)
        Link_cr_list.append(Link_tmp_cr1)
        
        return Link_ct_list, Link_cr_list

    def generate_class_tensors(self, branch, boxes, labels):
        """
        生成类别预测张量
        
        参数:
            branch (int): 分支索引 (0-3)
            boxes (Tensor): 边界框坐标
            labels (Tensor): 类别标签
            
        返回:
            tuple: (Q_list, Q_ct_list) 类别预测张量列表
        """
        Q_list = []
        Q_ct_list = []
        
        # 初始化张量
        Q_tensor = torch.zeros((self.s, self.s, self.classes))
        Q_ct_tensor = torch.zeros((self.s, self.s, self.classes))
        Q_tensor1 = torch.zeros((self.s, self.s, self.classes))
        Q_ct_tensor1 = torch.zeros((self.s, self.s, self.classes))
        
        # 为每个边界框设置类别预测
        for idx_ele, class_label in enumerate(labels):
            box = boxes[idx_ele]
            xmin, ymin, xmax, ymax = box
            
            # 防止坐标值过大
            if xmax >= 0.99:
                xmax -= 0.001
            if ymax >= 0.99:
                ymax -= 0.001
            
            # 根据分支索引设置不同角点的类别
            if branch == 0:  # 左下角
                Q_tensor[int(ymax * self.s), int(xmin * self.s), class_label] = self.infinite
                Q_tensor1[int(ymax * self.s), int(xmin * self.s), class_label] = self.infinite
            elif branch == 1:  # 左上角
                Q_tensor[int(ymin * self.s), int(xmin * self.s), class_label] = self.infinite
                Q_tensor1[int(ymin * self.s), int(xmin * self.s), class_label] = self.infinite
            elif branch == 2:  # 右下角
                Q_tensor[int(ymax * self.s), int(xmax * self.s), class_label] = self.infinite
                Q_tensor1[int(ymax * self.s), int(xmax * self.s), class_label] = self.infinite
            elif branch == 3:  # 右上角
                Q_tensor[int(ymin * self.s), int(xmax * self.s), class_label] = self.infinite
                Q_tensor1[int(ymin * self.s), int(xmax * self.s), class_label] = self.infinite
                
            # 设置中心点的类别
            center_y = int((ymin + ymax) / 2 * self.s)
            center_x = int((xmin + xmax) / 2 * self.s)
            Q_ct_tensor[center_y, center_x, class_label] = self.infinite
            Q_ct_tensor1[center_y, center_x, class_label] = self.infinite
        
        # 应用softmax激活
        Q_ct_tensor = F.softmax(Q_ct_tensor, dim=-1).clone()
        Q_tensor = F.softmax(Q_tensor, dim=-1).clone()
        Q_ct_tensor1 = F.softmax(Q_ct_tensor1, dim=-1).clone()
        Q_tensor1 = F.softmax(Q_tensor1, dim=-1).clone()
        
        Q_list.append(Q_tensor)
        Q_list.append(Q_tensor1)
        Q_ct_list.append(Q_ct_tensor)
        Q_ct_list.append(Q_ct_tensor1)
        
        return Q_list, Q_ct_list    #[y,x,Q]  (归一化后)

    def combine_tensors(self, Q_list, Q_ct_list, posi_list, posi_ct_list, 
                     Link_ct_list, Link_cr_list, pos_list, pos_ct_list):
        """
        合并所有特征张量，生成最终的PLN格式标签
        
        参数:
            Q_list: 角点类别张量列表
            Q_ct_list: 中心点类别张量列表
            posi_list: 角点存在性张量列表
            posi_ct_list: 中心点存在性张量列表
            Link_ct_list: 中心点链接张量列表
            Link_cr_list: 角点链接张量列表
            pos_list: 角点相对位置张量列表
            pos_ct_list: 中心点相对位置张量列表
            
        返回:
            Tensor: 形状为[14,14,204]的最终标签张量
        """
        list_feature = []
        list_ct_feature = []
        
        # 组合每个边界框的特征
        for i in range(self.B):
            # 组合中心点特征: [存在性, 相对位置, 连接关系, 类别]
            center_features = torch.cat((
                posi_ct_list[i], 
                pos_ct_list[i],
                Link_ct_list[i],
                Q_ct_list[i]
            ), dim=-1)
            
            # 组合角点特征: [存在性, 相对位置, 连接关系, 类别]
            corner_features = torch.cat((
                posi_list[i], 
                pos_list[i], 
                Link_cr_list[i], 
                Q_list[i]
            ), dim=-1)
            
            list_feature.append(corner_features)
            list_ct_feature.append(center_features)
        
        # 合并所有角点特征
        feature_tensor = torch.cat(list_feature, dim=-1)
        # 合并所有中心点特征
        feature_ct_tensor = torch.cat(list_ct_feature, dim=-1)
        
        # 合并中心点和角点特征，得到最终标签张量   #前102维中心点，后面102角点
        return torch.cat((feature_ct_tensor, feature_tensor), dim=-1)


if __name__ == '__main__':
    # 测试数据集加载
    device = 'cuda'
    file_root = r"E:\datasets\pascalvoc\pascalvoc\VOCdevkit\VOC2007\JPEGImages/"
    
    # 创建训练数据集
    train_dataset = PLNDataset(
        img_root=file_root, 
        list_file='voctrain.txt', 
        train=True,
        transform=[transforms.ToTensor()]
    )
    # 获取一个样本进行测试
    img, target = train_dataset[2]
    i= 1
    j = 9
    k = 8

    print(target[i,:,:,45])
    print("标签生成成功，形状:", target.shape)
