import random  # 随机数模块
import torch  # 导入pytorch
from d2l import torch as d2l  # 一些基本的深度学习函数


# 生成数据集
def synthetic_data(w, b, num_examples):  # 生成数据集
    """生成y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 随机生成 num_examples 行 len(w) 列的矩阵
    y = torch.matmul(X, w) + b  # 计算 y = Xw + b
    y += torch.normal(0, 0.01, y.shape)  # 加上噪声
    return X, y.reshape((-1, 1))  # 返回 X 和 转置后的 y -1自动将y的形状转为(num_examples,1)


# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)  # features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）
print('features:', features[0], '\nlabel:', labels[0])  # 打印第一个样本的特征和标签
# features: tensor([-1.0186, -0.1225])
# label: tensor([2.5836])


# 可视化数据集
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)  # 只有 detach 后才能转到 numpy 里面去


# 定义数据集读取器
def data_iter(batch_size, features, labels):  # 生成小批量数据集
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 对样本索引进行随机重排序，以便以每次迭代取出不同的数据子集
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个 batch_size，因此用 min 函数限制范围
        yield features[batch_indices], labels[batch_indices]  # yield 关键字用于生成迭代器，返回一个生成器对象，可以用 next() 函数获取下一个值


# 读取第一个小批量数据
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    """
    运行迭代时，会连续地获得不同的小批量，直至遍历完整个数据集
    循环一次，调用一次data_iter()
    每次迭代返回一个小批量的特征和标签
    再次调用时（由于yield关键字）从上次的位置继续
    """
    print(X, '\n', y)
    break
# tensor([[-0.1774,  0.9301],
#         [ 2.0726, -0.8996],
#         [-0.7972, -0.3161],
#         [ 0.0052, -1.3159],
#         [-0.1677,  1.6213],
#         [-0.7590,  1.0103],
#         [-0.9327,  1.1352],
#         [ 0.4481, -1.5346],
#         [ 0.4171,  0.0842],
#         [-0.5942, -1.3337]])
#  tensor([[ 0.6886],
#         [11.4139],
#         [ 3.6750],
#         [ 8.6796],
#         [-1.6448],
#         [-0.7698],
#         [-1.5164],
#         [10.3102],
#         [ 4.7424],
#         [ 7.5442]])

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # 随机初始化权重 requires_grad=True 表示需要对 w 进行求导
b = torch.zeros(1, requires_grad=True)  # 偏置初始化为 0 为标量


# 定义模型
def linreg(X, w, b):  # 线性回归模型
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):  # 均方损失函数
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # 对应元素的平方 除以 2 但未求均值


# 定义优化算法
def sgd(params, lr, batch_size):  # 小批量随机梯度下降算法
    with torch.no_grad():  # 在 torch.no_grad() 范围内，梯度不会被自动计算和累加 更新参数时不需要梯度计算
        for param in params:
            param -= lr * param.grad / batch_size  # 由于损失函数未计算均值，所以这里除以 batch_size
            param.grad.zero_()  # 梯度清零


# 训练模型
lr = 0.03  # 学习率
num_epochs = 3  # 迭代次数
# 方便替换
net = linreg  # 选择模型
loss = squared_loss  # 选择损失函数

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # 模型预测 计算损失
        # 因为l形状是(batch_size,1)，而不是一个标量
        # 需要调用.sum()函数来得到一个标量，并以此计算关于[w,b]的梯度
        l.sum().backward()  # 求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
    with torch.no_grad():  # 在 torch.no_grad() 范围内，梯度不会被自动计算和累加
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')  # 打印训练损失 mean()求均值
# epoch 1, loss 0.037162
# epoch 2, loss 0.000134
# epoch 3, loss 0.000051

print(f'w: {true_w}, b: {true_b}')  # 打印真实参数
print(f'w: {w.reshape((1,-1)), b}')  # 打印训练得到的参数
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
# w: tensor([ 2.0000, -3.4000]), b: 4.2
# w: (tensor([[ 1.9995, -3.3994]], grad_fn=<ViewBackward0>), tensor([4.1999], requires_grad=True))
