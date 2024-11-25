import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)  # 1000个样本


# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):  # 布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)  # 将数据和标签组合成一个数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)  # 返回的数据的迭代器
print(next(iter(data_iter)))  # iter(data_iter) 是一个迭代器对象，next是取迭代器里面的元素
# [tensor([[ 0.8482,  1.9892],
#         [-0.5166, -1.1064],
#         [ 2.2390,  0.2100],
#         [ 0.5310,  1.0847],
#         [-1.2754, -0.4711],
#         [ 1.4253, -0.9882],
#         [-2.3966,  0.4667],
#         [ 0.6666, -2.1511],
#         [-0.6097, -0.6963],
#         [-0.3293,  0.1273]]), tensor([[-0.8530],
#         [ 6.9264],
#         [ 7.9642],
#         [ 1.5826],
#         [ 3.2568],
#         [10.4177],
#         [-2.1703],
#         [12.8406],
#         [ 5.3467],
#         [ 3.1034]])]

# 定义模型
# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))  # 输入特征维度为2，输出维度为1

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # 权重参数初始化 0均值，0.01标准差
print(net[0].bias.data.fill_(0))  # 偏置参数初始化为0
# tensor([0.])

# 定义损失函数
loss = nn.MSELoss()  # 均方误差损失函数

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 随机梯度下降优化算法

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 计算损失
        trainer.zero_grad()  # 梯度清零
        l.backward()  # 反向传播
        trainer.step()  # 更新参数
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
# epoch 1, loss 0.000208
# epoch 2, loss 0.000100
# epoch 3, loss 0.000101

w = net[0].weight.data
b = net[0].bias.data
print(f'w: {w.numpy()}, b: {b.numpy()}')  # numpy()方法用于转换为numpy数组
# w: [[ 2.0004933 -3.399767 ]], b: [4.1998973]
