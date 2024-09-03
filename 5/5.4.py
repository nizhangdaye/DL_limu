import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
# tensor([-2., -1.,  0.,  1.,  2.])

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())
# tensor(1.8626e-09, grad_fn=<MeanBackward0>)

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))  # nn.Parameter使得这些参数加上了梯度
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


dense = MyLinear(5, 3)
print(dense.weight)
# Parameter containing:
# tensor([[-1.2242,  0.7284,  0.7146],
#         [ 1.0319,  0.7506,  1.4014],
#         [ 0.8037,  1.3523,  0.2606],
#         [ 0.3087, -0.8171, -0.4299],
#         [-0.2361, -0.4604, -0.0679]], requires_grad=True)

# 使用自定义层直接执行正向传播计算
print(dense(torch.rand(2, 5)))
# tensor([[0.8832, 0.0000, 0.0000],
#         [1.0869, 0.0000, 0.0000]])
# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
# tensor([[1.7961],
#         [2.0545]])