import torch
from d2l import torch as d2l
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256  # num_inputs:输入特征数，num_outputs:输出特征数，num_hiddens:隐藏层特征数

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)  # 权重参数初始化 行数为输入特征数，列数为隐藏层特征数
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))  # 偏置参数初始化
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)  # 权重参数初始化 行数为隐藏层特征数，列数为输出特征数
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))  # 偏置参数初始化

params = [W1, b1, W2, b2]  # 模型参数列表


# 实现 ReLU 激活函数
def relu(X):
    a = torch.zeros_like(X)  # 新建一个全零张量，形状和 X 相同
    return torch.max(X, a)  # 选择 X 和 全零张量中的最大值作为输出


# 实现模型
def net(X):
    X = X.reshape((-1, num_inputs))  # 将图像展平 改变 X 的形状为 (batch_size, num_inputs) 样本按行排列
    H = relu(X @ W1 + b1)  # 隐藏层输出 X @ W1 等价于 torch.matmul(X, W1)
    return H @ W2 + b2  # 输出层输出


# 定义交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')  # reduction='none' 表示返回每个样本的损失值

# 训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)  # 优化器
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 测试
d2l.predict_ch3(net, test_iter)