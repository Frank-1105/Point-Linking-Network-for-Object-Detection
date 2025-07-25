model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}
# 预测类别
CLASS_NUM = 20

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['InceptionResNetV2', 'inceptionresnetv2']

pretrained_settings = {
    'inceptionresnetv2': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 448, 448],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 20
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 448, 448],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 21
        }
    }

}


def expand_Cov():
    return nn.Sequential(
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=4, dilation=4),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=8, dilation=8),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=16, dilation=16),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=1),
        nn.BatchNorm2d(204),
        nn.ReLU(),
        nn.Conv2d(in_channels=204, out_channels=204, kernel_size=3, padding=1),
    )


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=20):
        super(InceptionResNetV2, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (448, 448, 3)
        self.mean = None
        self.std = None
        # Stem 448,448,3 --> 53,53,192
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()   # 53，53，320
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )  # 53，53，120
        self.mixed_6a = Mixed_6a()  # 26,26,1088
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        ) # 26,26,1088
        self.mixed_7a = Mixed_7a() # 12,12,2080
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(noReLU=True) # 12,12,2080
        # self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.conv2d_7b_yy = BasicConv2d(2080, 3328, kernel_size=1, stride=1, padding=1)   # 14,14,3328
        self.avgpool_1a = nn.AvgPool2d(1, count_include_pad=False)
        self.conv1 = BasicConv2d(3328, 1536, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(1536, 1536, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(1536, 1536, kernel_size=3, stride=1, padding=1)    # 14,14,1536
        # self.conv4 = BasicConv2d(1536, 1536, kernel_size=3, stride=1, padding=1)
        # self.conv5 = BasicConv2d(1536, 204, kernel_size=3, stride=1, padding=1)

        self.branches = nn.ModuleList()
        for _ in range(4):
            branch = nn.Sequential(
                nn.Conv2d(in_channels=1536, out_channels=1536, kernel_size=3, padding=1),
                nn.BatchNorm2d(1536),
                nn.ReLU(),
                nn.Conv2d(in_channels=1536, out_channels=204, kernel_size=3, padding=1),
                nn.BatchNorm2d(204),
                nn.ReLU(),
                expand_Cov(),
            )  # 14,14,204
            self.branches.append(branch)
        self.B = 2
        self.S = 14

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b_yy(x)
        x = self.avgpool_1a(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        return x

    '''def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
    '''

    def forward(self, input):
        # torch.Size([1, 1536, 14, 14])
        x = self.features(input)
        # x = x.permute(0, 2, 3, 1)
        branch_outputs = []
        for branch in self.branches:
            # torch.Size([1, 204, 14, 14])
            out = branch(x)
            out6_list = []
            for j in range(2 * self.B):
                # pij 0 xij yij 1-2 lxij 3-16 lyij 17-30 qij 31-50
                # 应该是对pij进行sigmoid
                out1 = torch.sigmoid(out[:,(23 + 2 * self.S) * j, ...]).unsqueeze(1)
                # ([batch, H, W]) -> ([batch,1, H, W])   204,14,14-->1,14,14   51*4=204
                # 对xij，yij进行sigmoid
                out2 = torch.sigmoid(
                    out[:, 1 + (23 + 2 * self.S) * j:3 + (23 + 2 * self.S) * j, ...])
                # Lx
                out3 = F.softmax((out[:, 3 + (23 + 2 * self.S) * j:(3 + self.S) + (23 + 2 * self.S) * j, ...]),
                                 dim=1).clone()
                # Ly
                out4 = F.softmax(
                    (out[:, (3 + self.S) + (23 + 2 * self.S) * j:(3 + 2 * self.S) + (23 + 2 * self.S) * j, ...]),
                    dim=1).clone()
                # Q
                out5 = F.softmax((out[:,(3 + 2 * self.S) + (23 + 2 * self.S) * j:(3 + 2 * self.S) + 20 + (23 + 2 * self.S) * j, ...]), dim=1).clone()
                # 排序顺序为P X Y Lx Ly Q
                out6_list.append(torch.cat((out1, out2, out3, out4, out5), dim=1))
            out = torch.cat(out6_list, dim=1)
            branch_outputs.append(out)
        return branch_outputs
        # return x


def inceptionresnetv2(num_classes=20, pretrained='imagenet'):
    r"""InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    if pretrained:
        settings = pretrained_settings['inceptionresnetv2'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionResNetV2(num_classes=20)
        model_dict = model.state_dict()
        # pretrained_dict = model_zoo.load_url(settings['url'])
        pretrained_dict = torch.load(r"E:\study\2025spring\baoyan\Paper reproduction\PLN\inceptionresnetv2-520b38e4.pth")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # if pretrained == 'imagenet':
        # new_last_linear = nn.Linear(1536, 1000)
        # new_last_linear.weight.data = model.last_linear.weight.data[1:]
        # new_last_linear.bias.data = model.last_linear.bias.data[1:]
        # model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']

        model.mean = settings['mean']
        model.std = settings['std']
    else:
        # print("mymodel")
        model = InceptionResNetV2(num_classes=num_classes)
    return model


def pretrained_inception():
    model = InceptionResNetV2(num_classes=20).to("cuda")
    model_dict = model.state_dict()
    # pretrained_dict = torch.load(r"/root/autodl-tmp/inceptionresnetv2-520b38e4.pth")
    pretrained_dict = torch.load(r"/root/autodl-tmp/inceptionresnetv2-520b38e4.pth")
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    assert inceptionresnetv2(num_classes=20, pretrained=None)
    # print(inceptionresnetv2(num_classes=20, pretrained=None))
    print('success1')
    assert inceptionresnetv2(num_classes=20, pretrained='imagenet')
    model = pretrained_inception().to("cuda")
    print(model)
    x = torch.randn((16, 3, 448, 448)).to("cuda")
    y = model(x)
    print(len(y))
    # y = y.permute(0, 2, 3, 1)
    print(y[3].size())
    print('success2')
