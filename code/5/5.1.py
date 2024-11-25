import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

X = torch.rand(2, 20)
print(net(X))


# tensor([[ 0.0819, -0.0157,  0.0066,  0.1114, -0.0334,  0.0681,  0.0723, -0.0512,
#          -0.0625, -0.0230],
#         [-0.0059, -0.1143,  0.1253,  0.1412,  0.0894,  0.1311,  0.0627, -0.0428,
#          -0.2419, -0.0045]], grad_fn=<AddmmBackward0>)


class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))


net = MLP()
print(net(X))


# tensor([[ 0.2562,  0.0372,  0.1668,  0.0516, -0.0890, -0.0652, -0.0237,  0.3862,
#           0.2786,  0.2802],
#         [ 0.2898,  0.1183,  0.1148,  0.0293,  0.1075, -0.0088, -0.1856,  0.3277,
#           0.1517,  0.2112]], grad_fn=<AddmmBackward0>)


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256),
                   nn.ReLU(),
                   nn.Linear(256, 10))
print(net(X))


# tensor([[-0.1429, -0.2546, -0.0147,  0.0478, -0.0973, -0.0653,  0.0362,  0.1126,
#           0.2140,  0.0528],
#         [-0.1651, -0.4223,  0.0700,  0.0756,  0.1032, -0.1010,  0.2794,  0.0988,
#           0.0485,  0.0246]], grad_fn=<AddmmBackward0>)

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
print(net(X))
# tensor(0.1110, grad_fn=<SumBackward0>)


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))
# tensor(-0.3325, grad_fn=<SumBackward0>)

