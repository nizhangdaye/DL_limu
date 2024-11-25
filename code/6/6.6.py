import torch
from torch import nn
from d2l import torch as d2l
import time

start = time.perf_counter()

# 定义LeNet
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

# 查看每层的输出形状
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
# Conv2d output shape: 	 torch.Size([1, 6, 28, 28])
# Sigmoid output shape: 	 torch.Size([1, 6, 28, 28])
# AvgPool2d output shape: 	 torch.Size([1, 6, 14, 14])
# Conv2d output shape: 	 torch.Size([1, 16, 10, 10])
# Sigmoid output shape: 	 torch.Size([1, 16, 10, 10])
# AvgPool2d output shape: 	 torch.Size([1, 16, 5, 5])
# Flatten output shape: 	 torch.Size([1, 400])
# Linear output shape: 	 torch.Size([1, 120])
# Sigmoid output shape: 	 torch.Size([1, 120])
# Linear output shape: 	 torch.Size([1, 84])
# Sigmoid output shape: 	 torch.Size([1, 84])
# Linear output shape: 	 torch.Size([1, 10])

# LeNet在Fashion-MNIST数据集上的表现
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


# 对evaluate_accuracy函数进行轻微的修改
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # net.eval()开启验证模式，不用计算梯度和更新梯度
        if not device:
            device = next(iter(net.parameters())).device  # 看net.parameters()中第一个元素的device为哪里
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]  # 如果X是个List，则把每个元素都移到device上
        else:
            X = X.to(device)  # 如果X是一个Tensor，则只用移动一次，直接把X移动到device上
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())  # y.numel() 为y元素个数
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""

    # 初始化模型参数
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)  # 使用Xavier初始化权重

    net.apply(init_weights)
    print('training on', device)
    net.to(device)  # 将模型参数移到device上
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 小批量随机梯度下降
    loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)  # 训练速度计时器
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()  # 切换到训练模式
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 梯度清零
            X, y = X.to(device), y.to(device)  # 移到device上
            y_hat = net(X)  # 前向传播计算预测值
            l = loss(y_hat, y)  # 计算损失
            l.backward()
            optimizer.step()  # 优化器更新参数
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:  # 每5个batch输出一次信息
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 在测试集上评价模型
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')  # 输出训练结果
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')  # 输出训练速度


lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# training on cuda:0
# loss 0.473, train acc 0.820, test acc 0.801
# 27967.0 examples/sec on cuda:0

end = time.perf_counter()
print(f"消耗时间：{end - start}秒")
# 消耗时间：81.1236003秒
