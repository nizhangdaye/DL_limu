import torch
from torch import nn
from d2l import torch as d2l


# 互相关运算
def corr2d(X, K):  # X 为输入，K为核矩阵
    """计算二维互相关信息"""
    h, w = K.shape  # 核矩阵的行数和列数
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))  # 输出图像的形状 由公式计算得出
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()  # 图片的小方块区域与卷积核做点积
    return Y


# 验证上述二维互相关运算的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


# tensor([[19., 25.],
#         [37., 43.]])


# 实现二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        self.weight = nn.Parameter(torch.rand(kernel_size))  # Parameter 用于定义可训练参数
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(Self, x):
        return corr2d(x, self.weight) + self.bias


# 卷积层的一个简单应用：检测图片中不同颜色的边缘
X = torch.ones((6, 8))
X[:, 2:6] = 0  # 把中间四列设置为0
print(X)  # 0 与 1 之间进行过渡，表示边缘

K = torch.tensor([[1.0, -1.0]])  # 如果左右原值相等，那么这两原值乘1和-1相加为0，则不是边缘
Y = corr2d(X, K)
print(Y)
print(corr2d(X.t(), K))  # X.t() 为X的转置，而K卷积核只能检测垂直边缘
# tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.]])
# tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
# tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])


# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)
# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2  # 损失函数
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
# epoch 2, loss 8.822
# epoch 4, loss 2.950
# epoch 6, loss 1.097
# epoch 8, loss 0.431
# epoch 10, loss 0.173

print(conv2d.weight.data.reshape((1, 2)))
# tensor([[ 0.9487, -1.0341]])

