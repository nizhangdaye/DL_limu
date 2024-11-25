import matplotlib.pyplot as plt
import torch
from IPython import display
from d2l import torch as d2l

# 获取数据集
bath_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(bath_size)  # 在源代码处加入了操作系统的判断

# 初始化模型参数
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


# 定义 softamx 操作
def softmax(X):
    """输入的 X 是一组预测值，输出 softmax 处理过的预测值"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)  # 列求和，保持维度
    return X_exp / partition  # 应用了广播机制


# 举例
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob)


# tensor([[0.1227, 0.2125, 0.2198, 0.1211, 0.3239],
#         [0.3685, 0.1344, 0.1431, 0.2738, 0.0803]])

# 定义模型
def net(X):
    """输入的 X 是图像数据，输出预测值"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)  # 使用reshape函数将每张原始图像展平为向量


# 定义损失函数
def cross_entropy(y_hat, y):
    """输入的 y_hat 是预测值，y 是真实标签，输出交叉熵损失"""
    return -torch.log(y_hat[range(len(y_hat)), y])  # 由于独热编码，只需计算实际标签对应的预测值的对数即可


# 举例 两个样本在三个类别中的概率
y = torch.tensor([0, 2])  # 实际标签索引
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # 两个样本在3个类别的预测概率
print(y_hat[[0, 1], y])  # # 把第0个样本对应标号"0"的预测值拿出来、第1个样本对应标号"2"的预测值拿出来
# tensor([0.1000, 0.5000])
print(cross_entropy(y_hat, y))


# tensor([2.3026, 0.6931])

# 分类精度
def accuracy(y_hat, y):
    """输入的 y_hat 是预测值，y 是真实标签，输出预算值与实际一致的个数"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # y_hat 为矩阵，y_hat.shape[1]>1表示不止一个类别，每个类别有各自的概率
        # print("y_hat_before", y_hat)
        # # y_hat_before tensor([[0.1000, 0.3000, 0.6000],
        # #         [0.3000, 0.2000, 0.5000]])
        y_hat = y_hat.argmax(axis=1)  # y_hat.argmax(axis=1)为求行最大值索引 axis=1表示按列求最大值
        # print("y_hat_after", y_hat)
        # # y_hat_after tensor([2, 2])
    cmp = y_hat.type(y.dtype) == y  # 先判断逻辑运算符==，再赋值给cmp，cmp为布尔类型的数据 type 函数用于转换数据类型
    return float(cmp.type(y.dtype).sum())  # 获得y.dtype的类型作为传入参数，将cmp的类型转为y的类型（int型），然后再求和


print("accuracy(y_hat,y) / len(y):", accuracy(y_hat, y) / len(y))
print("accuracy(y_hat,y):", accuracy(y_hat, y))  # 预测与实际一致的个数
print("len(y):", len(y))


# accuracy(y_hat,y) / len(y): 0.5
# accuracy(y_hat,y): 1.0
# len(y): 2

# 可以评估在任意模型net的准确率
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):  # 如果net模型是torch.nn.Module实现的神经网络的话，将它变成评估模式
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数，metric为累加器的实例化对象，里面存了两个数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())  # net(X)将X输入模型，获得预测值。y.numel()为样本总数
    return metric[0] / metric[1]  # 分类正确的样本数 / 总样本数


# Accumulator实例中创建了2个变量，用于分别存储正确预测的数量和预测的总数量
class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0, 0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # zip函数把两个列表第一个位置元素打包、第二个位置元素打包....

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


print(evaluate_accuracy(net, test_iter))  # 输出模型在测试集上的精度


# 0.0994

# 训练模型
def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    if isinstance(net, torch.nn.Module):  # 如果net模型是torch.nn.Module实现的神经网络的话，将它变成训练模式
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):  # 如果updater是torch.optim.Optimizer实例的话，使用PyTorch内置的优化器和损失函数
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()  # 梯度清零
            l.mean().backward()  # 计算梯度
            updater.step()  # 优化并更新参数
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """
        初始化Animator对象

        参数:
        - xlabel: x轴标签
        - ylabel: y轴标签
        - legend: 图例
        - xlim: x轴范围
        - ylim: y轴范围
        - xscale: x轴比例
        - yscale: y轴比例
        - fmts: 线条格式
        - nrows: 子图行数
        - ncols: 子图列数
        - figsize: 图表大小
        """
        if legend is None:
            legend = []
        # 使用SVG显示动画
        d2l.use_svg_display()
        # d2l.plt.ion()  # 开启交互模式
        # 创建包含多个子图的图表
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        # 如果只有一个子图，则将其放入列表中，方便后续处理
        if nrows * ncols == 1:
            self.axes = [self.axes]
        # 使用lambda函数配置坐标轴
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        # 初始化X和Y坐标以及线条格式
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """
        向图表中添加多个数据点

        参数:
        - x: x坐标数据
        - y: y坐标数据
        """
        # 如果y不是列表或数组，将其转换为列表
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        # 如果x不是列表或数组，将其转换为与y相同长度的列表
        if not hasattr(x, "__len__"):
            x = [x] * n
        # 如果X和Y坐标为空，则初始化它们
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        # 将数据添加到X和Y坐标中
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        # 清空图表并绘制线条
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        # 配置坐标轴并显示图表
        self.config_axes()
        d2l.plt.draw()  # 更新图表
        d2l.plt.pause(0.001)  # 短暂暂停以更新图表


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train_loss', 'train_acc', 'test_acc'])  # 绘制动画
    for epoch in range(num_epochs):  # 训练num_epochs个迭代周期
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)  # 训练一个迭代周期
        test_acc = evaluate_accuracy(net, test_iter)  # 在测试集上评估模型
        animator.add(epoch + 1, train_metrics + (test_acc,))  # 向动画中添加数据
        train_loss, train_acc = train_metrics
        print(f'num_epochs {epoch + 1}, train_loss {train_loss:.3f}, train_acc {train_acc:.3f}')
        # num_epochs 1, train_loss 0.786, train_acc 0.751
        # num_epochs 2, train_loss 0.569, train_acc 0.813
        # num_epochs 3, train_loss 0.524, train_acc 0.826
        # num_epochs 4, train_loss 0.500, train_acc 0.833
        # num_epochs 5, train_loss 0.486, train_acc 0.836
        # num_epochs 6, train_loss 0.474, train_acc 0.840
        # num_epochs 7, train_loss 0.465, train_acc 0.842
        # num_epochs 8, train_loss 0.458, train_acc 0.846
        # num_epochs 9, train_loss 0.453, train_acc 0.847
        # num_epochs 10, train_loss 0.446, train_acc 0.849
    assert train_loss < 0.5, train_loss  # 训练损失不应超过0.5
    assert train_acc <= 1 and train_acc > 0.7, train_acc  # 训练精度应介于0.7和1之间
    assert test_acc <= 1 and test_acc > 0.7, test_acc  # 测试精度应介于0.7和1之间


lr = 0.1  # 学习率


# 优化函数
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)  # 使用小批量随机梯度下降


num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=10):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:  # 测试数据集
        break
    trues = d2l.get_fashion_mnist_labels(y)  # 获得真实标签
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))  # 获得预测标签
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


predict_ch3(net, test_iter)
