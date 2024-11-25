import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import time

start = time.perf_counter()

# 获取数据集
# @save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                          'fba480ffa8aa7e0febbb511d181409f899b9baa5')  # 下载数据集
data_dir = d2l.download_extract('hotdog')  # 下载解压数据集

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))  # 读取训练集图片
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))  # 读取测试集图片

hotdogs = [train_imgs[i][0] for i in range(8)]  # 前8张热狗图片
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]  # 后8张非热狗图片
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)  # 显示热狗和非热狗图片

# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet数据集的均值和标准差

# 数据增广
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),  # 随机裁剪为源图像的大小
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
    torchvision.transforms.ToTensor(),  # 转换为Tensor
    normalize])  # 标准化

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),  # 调整图像大小为256×256
    torchvision.transforms.CenterCrop(224),  # 裁剪图像中央的224×224
    torchvision.transforms.ToTensor(),
    normalize])

# 定义和初始化模型
pretrained_net = torchvision.models.resnet18(pretrained=True)  # 预训练的ResNet-18模型  pretrained: 是否加载预训练模型的参数
print(pretrained_net)  # 打印模型结构
# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=1000, bias=True)
# )
print(pretrained_net.fc)  # 打印输出层
# Linear(in_features=512, out_features=1000, bias=True)
finetune_net = torchvision.models.resnet18(pretrained=True)  # 微调的ResNet-18模型
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)  # 重新定义输出层 2个类别
nn.init.xavier_uniform_(finetune_net.fc.weight)  # 使用Xavier初始化输出层(最后一层)


# 微调模型
# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")  # reduction: 损失函数的计算方式，none表示返回每个样本的损失值
    if param_group:
        """
        除了最后一层的learning rate外，用的是默认的learning rate
        最后一层的learning rate用的是十倍的learning rate
        """
        params_1x = [param for name, param in net.named_parameters()  # named_parameters: 网络中的参数
                     if name not in ["fc.weight", "fc.bias"]]  # 不包括最后一层的模型参数
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(), 'lr': learning_rate * 10}],  # params: 要优化的参数
                                  # 因为最后一层是随机初始化的，希望它学习的更快，所以用十倍的学习率
                                  lr=learning_rate, weight_decay=0.001)  # weight_decay: 正则化参数
    else:  # 不使用源模型的参数组
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


# 训练微调模型
train_fine_tuning(finetune_net, 5e-5)  # 学习率很小
# loss 0.223, train acc 0.924, test acc 0.934
# 1879.0 examples/sec on [device(type='cuda', index=0)]

# 保存模型参数
try:
    torch.save(finetune_net.state_dict(), 'resnet18_13.2.params')
    print("模型参数已保存。")
except Exception as e:
    print(f"保存模型参数时出错: {e}")

# 训练一个从头开始的模型
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)  # 重新定义输出层 2个类别
train_fine_tuning(scratch_net, 5e-4, param_group=False)  # 学习率较大
# loss 0.366, train acc 0.839, test acc 0.804
# 2086.2 examples/sec on [device(type='cuda', index=0)]

end = time.perf_counter()
print(f"Time taken: {end - start:.2f} seconds")
