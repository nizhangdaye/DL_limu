import torch
from torch import nn
from d2l import torch as d2l



# 初始化模型参数
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状 保存第 0 维 其他维度展成 1 维张量
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))  # sequential 将多个层组合成一个网络 flattern将输入数据展平


def init_weights(m):
    if type(m) == nn.Linear:  # 判断 m 是否为线性层
        nn.init.normal_(m.weight, std=0.01)  # 使用正态分布初始化权重


net.apply(init_weights)  # 应用初始化函数

batch_size, lr, num_epochs = 256, 0.1, 10

# 定义交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')  # 定义损失函数，reduction='none'表示返回每个样本的损失值

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 加载数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 训练模型
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

