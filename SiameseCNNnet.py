'''
参考搭建VGG网络：https://blog.csdn.net/muye_IT/article/details/123927270
'''

from torch.hub import load_state_dict_from_url
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms
import torch.nn.functional as F

'''
SiameseVGG
'''
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 512),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 105, 105, 3   -> 105, 105, 64 -> 52, 52, 64
# 52, 52, 64    -> 52, 52, 128  -> 26, 26, 128
# 26, 26, 128   -> 26, 26, 256  -> 13, 13, 256
# 13, 13, 256   -> 13, 13, 512  -> 6, 6, 512
# 6, 6, 512     -> 6, 6, 512    -> 3, 3, 512
def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# vgg网络模型配置列表，数字表示卷积核个数，'M'表示最大池化层
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],											# 模型A
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],									# 模型B
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],					# 模型D
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], 	# 模型E
}

def VGG16(pretrained, in_channels, **kwargs):
    model = VGG(make_layers(cfgs["vgg11"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    return model

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height)

class SiameseVGG(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(SiameseVGG, self).__init__()
        self.vgg = VGG16(pretrained, 3)
        # del self.vgg.avgpool
        # del self.vgg.classifier

        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1 = self.vgg.features(x1)
        x2 = self.vgg.features(x2)
        # -------------------------#
        #   相减取绝对值，取l1距离
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        x1 = self.fully_connect1(x1)
        x2 = self.fully_connect1(x2)

        x = torch.abs(x1[0] - x2[0])
        # -------------------------#
        #   进行两次全连接
        # -------------------------#

        x = self.fully_connect2(x)

        return x, x1, x2

'''
SiameseResNet18
'''

transform = transforms.Compose([ToTensor(),
                                transforms.Normalize(
                                    mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5]
                                ),
                                transforms.Resize((224, 224))
                                ])


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 原地替换 节省内存开销
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        #         print('shape of x: {}'.format(x.shape))
        out = self.layer(x)
        #         print('shape of out: {}'.format(out.shape))
        #         print('After shortcut shape of x: {}'.format(self.shortcut(x).shape))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock, 64, [[1, 1], [1, 1]])
        # self.conv2_2 = self._make_layer(BasicBlock,64,[1,1])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        # self.conv3_2 = self._make_layer(BasicBlock,128,[1,1])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock, 256, [[2, 1], [1, 1]])
        # self.conv4_2 = self._make_layer(BasicBlock,256,[1,1])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]])
        # self.conv5_2 = self._make_layer(BasicBlock,512,[1,1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 512)

    # 这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        #         out = F.avg_pool2d(out,7)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height)

class SiameseResNet18(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(SiameseResNet18, self).__init__()
        self.resnet = ResNet18()
        # del self.resnet.avgpool
        # del self.resnet.fc

        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        # 这一行，那个卷积层的输入参数要注意，暂时还不知道怎么得来的
        self.fully_connect1 = torch.nn.Linear(512, 1)
        # self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1 = self.resnet(x1)
        x2 = self.resnet(x2)
        # -------------------------#
        #   相减取绝对值，取l1距离
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1[0] - x2[0])
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x = self.fully_connect1(x)
        # x = self.fully_connect2(x)

        return x, x1, x2

'''
SiameseResNet150
'''
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], first=False) -> None:
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 原地替换 节省内存开销
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 原地替换 节省内存开销
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        # if stride[1] != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         # 卷积核为1 进行升降维
        #         # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
        #         nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride[1], bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet50(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(ResNet50, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # conv2
        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)

        # conv3
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)

        # conv4
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)

        # conv5
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 512)

    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        # 用来判断是否为每个block层的第一层
        flag = True
        for i in range(0, len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

class SiameseResNet50(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(SiameseResNet50, self).__init__()
        self.resnet = ResNet50()
        # del self.resnet.avgpool
        # del self.resnet.fc

        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        # 这一行，那个卷积层的输入参数要注意，暂时还不知道怎么得来的
        self.fully_connect1 = torch.nn.Linear(512, 1)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1 = self.resnet(x1)
        x2 = self.resnet(x2)
        # -------------------------#
        #   相减取绝对值，取l1距离
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1[0] - x2[0])
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x = self.fully_connect1(x)
        # x = self.fully_connect2(x)

        return x, x1, x2


'''
SiameseResNet101

https://blog.csdn.net/TTTSEP9TH2244/article/details/123123388
'''

class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.down(x)
        return out

class ResNet101(nn.Module):
    def __init__(self):  # 指定分类数
        super(ResNet101, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # --------------------------------------------------------------------
        self.layer1_first = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256)
        )
        self.layer1_next = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256)
        )
        # --------------------------------------------------------------------
        self.layer2_first = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512)
        )
        self.layer2_next = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512)
        )
        # --------------------------------------------------------------------
        self.layer3_first = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.layer3_next = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024)
        )
        # --------------------------------------------------------------------
        self.layer4_first = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048)
        )
        self.layer4_next = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048)
        )
        # --------------------------------------------------------------------
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048 * 1 * 1, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 512)
        )

    def forward(self, x):
        out = self.pre(x)
        # --------------------------------------------------------------------
        layer1_shortcut = DownSample(64, 256, 1)
        layer1_shortcut.to('cuda:0')
        layer1_identity = layer1_shortcut(out)
        out = self.layer1_first(out)
        out = F.relu(out + layer1_identity, inplace=True)

        for i in range(2):
            identity = out
            out = self.layer1_next(out)
            out = F.relu(out + identity, inplace=True)
        # --------------------------------------------------------------------
        layer2_shortcut = DownSample(256, 512, 2)
        layer2_shortcut.to('cuda:0')
        layer2_identity = layer2_shortcut(out)
        out = self.layer2_first(out)
        out = F.relu(out + layer2_identity, inplace=True)

        for i in range(3):
            identity = out
            out = self.layer2_next(out)
            out = F.relu(out + identity, inplace=True)
        # --------------------------------------------------------------------
        layer3_shortcut = DownSample(512, 1024, 2)
        layer3_shortcut.to('cuda:0')
        layer3_identity = layer3_shortcut(out)
        out = self.layer3_first(out)
        out = F.relu(out + layer3_identity, inplace=True)

        for i in range(22):
            identity = out
            out = self.layer3_next(out)
            out = F.relu(out + identity, inplace=True)
        # --------------------------------------------------------------------
        layer4_shortcut = DownSample(1024, 2048, 2)
        layer4_shortcut.to('cuda:0')
        layer4_identity = layer4_shortcut(out)
        out = self.layer4_first(out)
        out = F.relu(out + layer4_identity, inplace=True)

        for i in range(2):
            identity = out
            out = self.layer4_next(out)
            out = F.relu(out + identity, inplace=True)
        # --------------------------------------------------------------------
        out = self.avg_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out

class SiameseResNet101(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(SiameseResNet101, self).__init__()
        self.resnet = ResNet101()
        # del self.resnet.avgpool
        # del self.resnet.fc

        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        # 这一行，那个卷积层的输入参数要注意，暂时还不知道怎么得来的
        self.fully_connect1 = torch.nn.Linear(512, 1)
        # self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1 = self.resnet(x1)
        x2 = self.resnet(x2)
        # -------------------------#
        #   相减取绝对值，取l1距离
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1[0] - x2[0])
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x = self.fully_connect1(x)
        # x = self.fully_connect2(x)

        return x, x1, x2

