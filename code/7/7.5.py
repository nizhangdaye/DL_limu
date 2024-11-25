import torch
from torch import nn
from d2l import torch as d2l
import time

start = time.perf_counter()


# 定义批量归一化函数
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):  # X为输入，gamma、beta为学的参数。moving_mean、moving_var为全局的均值、方差。eps为避免除0的参数。momentum为更新moving_mean、moving_var的
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 'is_grad_enabled' 来判断当前模式是训练模式还是预测模式。就是在做推理的时候，推理不需要反向传播，所以不需要计算梯度
        X_hat = (X - moving_mean) / torch.sqrt(
            moving_var + eps)  # 做推理时，可能只有一个图片进来，没有一个批量进来，因此这里用的全局的均值、方差。在预测中，一般用整个预测数据集的均值和方差。加eps为了避免方差为0，除以0了
    else:
        assert len(X.shape) in (2, 4)  # 批量数+通道数+图片高+图片宽=4
        if len(X.shape) == 2:  # 2 表示有两个维度，批量大小和特征
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)  # 按行求均值，即对每个特征维求一个均值
            var = ((X - mean) ** 2).mean(dim=0)
        else:  # == 4 表示有四个维度，批量数、通道数、图片高、图片宽
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)  # 沿通道维度求均值
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean  # 累加，将计算的均值累积到全局的均值上，更新moving_mean momentum 一般为0.9，即前一轮的均值和当前的均值做加权平均。
        moving_var = momentum * moving_var + (1.0 - momentum) * var  # 当前全局的方差与当前算的方差做加权平均，最后会无限逼近真实的方差。仅训练时更新，推理时不更新
    Y = gamma * X_hat + beta  # Y 为归一化后的输出
    return Y, moving_mean.data, moving_var.data


# 批量归一化层
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:  # 全连接层的情况
            shape = (1, num_features)
        else:  # 卷积层的情况
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        # 伽马、贝塔需要在反向传播时更新，所以放在nn.Parameter里面，moving_mean、moving_var不需要迭代，所以不放在里面
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# 在 LeNet 网络中使用批量归一化层
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),  ## BatchNorm 的特征维度为 6 即为通道
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))
# loss 0.266, train acc 0.901, test acc 0.853
# 25011.8 examples/sec on cuda:0
# tensor([2.8180, 2.9079, 1.5509, 2.0058, 4.7415, 3.1875], device='cuda:0',
#        grad_fn=<ViewBackward0>)
# tensor([-3.1075,  2.9179, -1.1533,  1.9685, -1.7691,  1.3454], device='cuda:0',
#        grad_fn=<ViewBackward0>)
# Time taken: 86.5487827 seconds

end = time.perf_counter()
print(f"Time taken: {end - start} seconds")

# 使用框架
start = time.perf_counter()

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# loss 0.265, train acc 0.902, test acc 0.863
# 27979.6 examples/sec on cuda:0
# Time taken: 85.2087728 seconds

end = time.perf_counter()
print(f"Time taken: {end - start} seconds")
