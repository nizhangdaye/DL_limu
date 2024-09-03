import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import time

start_time = time.perf_counter()

d2l.set_figsize()
img = d2l.Image.open('cat1.jpg')  # 读取图片
d2l.plt.imshow(img)  # 显示图片
d2l.plt.show()


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    """
    应用图像增广，并展示结果
    """
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()


# 翻转和裁剪
apply(img, torchvision.transforms.RandomHorizontalFlip())  # 水平翻转, 50%的概率
apply(img, torchvision.transforms.RandomVerticalFlip())  # 垂直翻转, 50%的概率
shape_aug = torchvision.transforms.RandomResizedCrop(  # 裁剪并调整大小
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))  # 裁剪尺寸为200x200, 缩放范围为(0.1, 1), 长宽比范围为(0.5, 2)
apply(img, shape_aug)

# 改变颜色
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))  # 亮度变化, 随机值为原始图像的50%到150%之间
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))  # 色调变化, 50%的概率
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)  # 同时设置亮度、对比度、饱和度、色调变化的概率
apply(img, color_aug)

# 结合多个增广
augs = torchvision.transforms.Compose([  # Compose将多个增广组合起来
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

# 使用图像增广进行训练
all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=False)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);  # 显示前32张训练集图片
d2l.plt.show()

# 定义训练集和测试集的图像增广
train_augs = torchvision.transforms.Compose([  # 训练集增广
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([  # 测试集增广
    torchvision.transforms.ToTensor()])


# 定义数据集
def load_cifar10(is_train, augs, batch_size):
    """
    加载CIFAR-10数据集, 并应用图像增广
    """
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader


# 定义训练函数
# @save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多GPU进行小批量训练"""
    if isinstance(X, list):  # isinstance()函数来判断对象是否是一个已知的类型
        # 微调BERT中所需
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()  # 训练模式
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()  # step()函数用来更新权重
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


# @save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用多GPU进行模型训练"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])  # DataParallel用于并行计算
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):  # 遍历训练集 features是图像, labels是标签 小批量
            timer.start()
            l, acc = train_batch_ch13(  # 训练一个小批量
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:  # 每5个batch输出一次信息
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)  # 计算测试集准确度
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
              f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)


def init_weights(m):
    """
    初始化模型权重
    """
    if type(m) in [nn.Linear, nn.Conv2d]:  # 全连接层和卷积层的权重初始化
        nn.init.xavier_uniform_(m.weight)  # 使用Xavier初始化方法


net.apply(init_weights)


def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    """
    训练模型，使用图像增广
    """
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")  # 交叉熵损失函数
    trainer = torch.optim.Adam(net.parameters(), lr=lr)  # Adam优化器算是一个比较平滑的SGD，它对学习率调参不是很敏感
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


train_with_data_aug(train_augs, test_augs, net)
# loss 0.162, train acc 0.944, test acc 0.807
# 9049.0 examples/sec on [device(type='cuda', index=0)]
# 总耗时：128.05秒


# 保存模型参数
try:
    torch.save(net.state_dict(), 'resnet18_13.1.params')
    print("模型参数已保存。")
except Exception as e:
    print(f"保存模型参数时出错: {e}")

end_time = time.perf_counter()
print(f"总耗时：{end_time - start_time:.2f}秒")
