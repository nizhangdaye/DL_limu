import torch

x = torch.arange(4.0)
print(x)
# tensor([0., 1., 2., 3.])

x.requires_grad_(True)  # 等价于 x = torch.arange(4.0, requires_grad=True)，将 x 标记为需要求导
print(x.grad)  # 默认值是None
# None

y = 2 * torch.dot(x, x)  # 2*x.T*x
print(y)
# tensor(28., grad_fn=<MulBackward0>)

y.backward()  # 计算梯度
print(x.grad)
# tensor([ 0.,  4.,  8., 12.])
print(x.grad == 4 * x)
# tensor([True, True, True, True])

# 在默认情况下，PyTorch会累积梯度，需要清除之前的值
x.grad.zero_()  # 清空梯度
y = x.sum()
y.backward()
print(x.grad)
# tensor([1., 1., 1., 1.])

x.grad.zero_()  # 清空梯度
y = x * x
y.sum().backward()  # 等价于 y.backward(torch.ones(y)) 将 y 转化为标量，计算 y 对 x 的导数
print(x.grad)
# tensor([0., 2., 4., 6.])

x.grad.zero_()  # 清空梯度
y = x * x
u = y.detach()  # 切断梯度流，u 不会影响 x 的梯度计算
z = u * x  # 将 u 作为常数项参与运算
z.sum().backward()
print(x.grad == u)
# tensor([True, True, True, True])

x.grad.zero_()  # 清空梯度
y.sum().backward()
print(x.grad == 2 * x)
# tensor([True, True, True, True])

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)  # 随机初始化参数，需要求导，形状为()，即标量
d = f(a)
d.backward()
print(a.grad == d / a)
# tensor(True)

import torch
import matplotlib.pyplot as plt
import numpy as np
x = torch.linspace(0, 3*np.pi, 128)  # 生成一个包含128个点的张量，范围从0到3π
x.requires_grad_(True)  # 标记张量 x 需要计算梯度
y = torch.sin(x)  # 计算张量 x 中每个元素的正弦值，得到新的张量 y
# 对 y 的所有元素求和，并进行反向传播计算梯度
y.sum().backward()
# 绘制 sin(x) 曲线和其导数曲线 ∂y/∂x=cos(x)，并在图例中给出相应的标签
plt.plot(x.detach(), y.detach(), label='y=sin(x)')
plt.plot(x.detach(), x.grad, label='∂y/∂x=cos(x)')  # dy/dx = cos(x)
# 添加图例并显示图形
plt.xticks(np.arange(0, 3*np.pi + np.pi, np.pi), ['0', 'π', '2π', '3π'])
plt.legend(loc='upper right')
plt.show()

