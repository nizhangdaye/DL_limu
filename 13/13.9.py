import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

pretrained_net = torchvision.models.resnet18(pretrained=True)
print(list(pretrained_net.children())[-3:])  # 展示预训练模型的最后三层
# [Sequential(
#   (0): BasicBlock(
#     (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (downsample): Sequential(
#       (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (1): BasicBlock(
#     (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   )
# ),
# AdaptiveAvgPool2d(output_size=(1, 1)),
# Linear(in_features=512, out_features=1000, bias=True)]

# 创建一个全卷积网络实例net
net = nn.Sequential(*list(pretrained_net.children())[:-2])  # 去掉ResNet18最后两层
X = torch.rand(size=(1, 3, 320, 480))
# 给定高度为320和宽度为480的输入，net的前向传播将输入的高和宽减小至原来的，即10和15
print(net(X).shape)
# torch.Size([1, 512, 10, 15])

num_classes = 21
# 使用 1*1 卷积层将输出通道数转换为Pascal VOC2012数据集的类数（21类）
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
# 使用转置卷积层将输出的高和宽恢复至输入的高和宽，并得到类别预测结果
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                    kernel_size=64, padding=16, stride=32))


# 初始化转置卷积层 双线性插值核的实现
def bilinear_kernel(in_channels, out_channels, kernel_size):
    """
    返回一个初始化的双线性插值核权重矩阵，可用于初始化转置卷积层
    """
    # 计算双线性插值核中心点位置
    factor = (kernel_size + 1) // 2
    # 根据核的大小是奇数还是偶数，确定中心点的位置
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    # 创建一个矩阵，其元素的值等于其与中心点的距离
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    # 计算双线性插值核，其值由中心点出发，向外线性衰减
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    # 初始化一个权重矩阵，大小为 (输入通道数, 输出通道数, 核大小, 核大小)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    # 将双线性插值核的值赋给对应位置的权重
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

img = torchvision.transforms.ToTensor()(d2l.Image.open('catdog.jpg'))
X = img.unsqueeze(0)  # 增加批量维度
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()  # 输出图像的维度是(H, W, C) Y[0] 是第一个样本

d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
# input image shape: torch.Size([561, 728, 3])
d2l.plt.imshow(img.permute(1, 2, 0))
print('output image shape:', out_img.shape)
# output image shape: torch.Size([1122, 1456, 3])
d2l.plt.imshow(out_img)

# 对最后一层的权重进行初始化
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# 读取数据集
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)


# read 1114 examples
# read 1078 examples


# 训练
# 交叉熵损失函数
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)


num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


# loss 0.446, train acc 0.861, test acc 0.854
# 107.1 examples/sec on [device(type='cuda', index=0), device(type='cuda', index=1)]

# 预测
def predict(img):
    """
    定义预测函数，输入参数img是待预测的图像
    :param img:
    :return:
    """
    # 首先，对图像进行归一化处理，并添加一个批量维度，以匹配模型的输入需求
    # normalize_image函数会对图像的每个像素进行归一化处理，使其值在0到1之间
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    # 使用argmax(dim=1)找出预测结果中概率最大的类别，返回这个类别的索引
    pred = net(X.to(devices[0])).argmax(dim=1)  # pred 是个标量，不是向量
    # 最后，将预测结果的形状改变成与原始图像相同的形状
    return pred.reshape(pred.shape[1], pred.shape[2])


# 可视化预测的类别
def label2image(pred):
    # 使用VOC_COLORMAP将类别转换为RGB颜色
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])  # 把类别的RGB值做成一个tensor
    # 将预测的结果转换为long型以对应颜色映射的索引
    X = pred.long()  # 转换成long型，以对应colormap的索引
    return colormap[X, :]  # 根据索引取出RGB值，并拼接成图像


voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    # 设置裁剪的区域
    crop_rect = (0, 0, 320, 480)
    # 对图像进行裁剪
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    # 预测裁剪后的图像，并将预测结果转换为图像
    pred = label2image(predict(X))  # 预测转成图片
    # 将原图，预测的图像和标签图像加入到imgs列表中
    imgs += [X.permute(1, 2, 0), pred.cpu(),
             torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1, 2, 0)]
# 显示原图、预测的图像和标签图像
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)  # 第二行为预测，第三行为真实标号
