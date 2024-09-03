import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))
# tensor([[0.1788],
#         [0.1728]], grad_fn=<AddmmBackward0>)

# 参数访问

print(net[2].state_dict())  # # 访问参数，net[2]就是最后一个输出层
# OrderedDict([('weight', tensor([[ 0.1890,  0.2337,  0.1736,  0.2461, -0.2516,  0.0174,  0.2657, -0.2497]])), ('bias', tensor([0.2529]))])
# 访问目标参数
print(type(net[2].bias))  # 访问bias参数类型
# <class 'torch.nn.parameter.Parameter'>
print(net[2].bias)  # 访问bias参数值
# Parameter containing:
# tensor([0.2826], requires_grad=True)
print(net[2].bias.data)  # 访问bias参数值，除去其他信息
# tensor([0.2826])
# 访问梯度
print(net[2].weight.grad == None)  # 由于还未进行反向传播，grad为None
# True


#  一次性访问所有参数

print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # *表示解包
# ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
print(*[(name, param.shape) for name, param in net.named_parameters()])  # 0是第一层名字，1是ReLU，它没有参数
# ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
print(net.state_dict()['2.bias'].data)  # 通过名字获取参数


# tensor([-0.3512])


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
# tensor([[0.0103],
#         [0.0104]], grad_fn=<AddmmBackward0>)

print(rgnet)
# Sequential(
#   (0): Sequential(
#     (block 0): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 1): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 2): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 3): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#   )
#   (1): Linear(in_features=4, out_features=1, bias=True)
# )

# 访问嵌套的层的参数
print(rgnet[0][1][0].bias.data)


# tensor([0.2691, 0.0448, 0.3411, 0.2580, 0.2636, 0.2124, 0.2467, 0.1614])


# 参数初始化

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 下划线表示把m.weight的值替换掉
        nn.init.zeros_(m.bias)


net.apply(init_normal)  # 会递归调用 直到所有层都初始化
print(net[0].weight.data[0], net[0].bias.data[0])


# tensor([-0.0017, -0.0012,  0.0155,  0.0155]) tensor(0.)


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print(net[0].weight.data[0])
print(net[0].bias.data[0])


# tensor([1., 1., 1., 1.])
# tensor(0.)


# 对某些块应用不同的初始化
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)


# tensor([-0.0078,  0.1388, -0.4263,  0.5073])
# tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])


# 自定义初始化函数
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])  # 打印名字是啥，形状是啥
        nn.init.uniform_(m.weight, -10, 10)  # 均匀分布初始化
        m.weight.data *= m.weight.data.abs() >= 5  # 大于5的权重乘以5 先判断是否大于等于5得到布尔矩阵


net.apply(my_init)
# Init weight torch.Size([8, 4])
# Init weight torch.Size([1, 8])
print(net[0].weight[:2])
# tensor([[-0.0000, 0.0000, 0.0000, 8.4138],
#         [-0.0000, 0.0000, 0.0000, 9.4350]], grad_fn=<SliceBackward0>)

# 直接替换参数
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])
# tensor([42.0000,  1.0000,  1.0000,  9.4138])

# 共享参数

# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
# tensor([True, True, True, True, True, True, True, True])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
# tensor([True, True, True, True, True, True, True, True])
