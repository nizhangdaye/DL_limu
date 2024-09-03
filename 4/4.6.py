import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    """
    丢弃法激活函数
    :param X: 输入数据
    :param dropout: 丢弃率
    """
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()  # 随机生成一个0-1的mask
    return mask * X / (1.0 - dropout)  # 乘以mask，将mask为0的元素置为0，将mask为1的元素乘以1/(1-dropout)


X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
print(X)
# tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
print(dropout_layer(X, 0.))
# tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
print(dropout_layer(X, 0.5))
# tensor([[ 0.,  0.,  0.,  0.,  0., 10.,  0.,  0.],
#         [16.,  0.,  0., 22., 24.,  0.,  0., 30.]])
print(dropout_layer(X, 1.))
# tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.]])


# 定义模型
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        # 第一层隐藏层
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        # 第二层隐藏层
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        # 输出层 未经过softmax函数
        out = self.lin3(H2)
        return out


# 训练和测试模型
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
num_epochs, lr, batch_size = 10, 0.5, 256
dropout1, dropout2 = 0.2, 0.5
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
loss = nn.CrossEntropyLoss(reduction='none')  # 交叉熵损失函数 用于多类分类
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 加载数据集
trainer = torch.optim.SGD(net.parameters(), lr=lr)  # 梯度下降优化器
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)  # 训练模型

# 简洁实现
net = nn.Sequential(nn.Flatten(),  # 展平输入
                    nn.Linear(784, 256),  # 第一层隐藏层
                    nn.ReLU(),
                    # 在第一层隐藏层之后添加一个dropout层
                    nn.Dropout(dropout1),
                    nn.Linear(256, 256),  # 第二层隐藏层
                    nn.ReLU(),
                    # 在第二层隐藏层之后添加一个dropout层
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10))  # 输出层


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)  # 使用正态分布初始化权重 0.01为标准差


net.apply(init_weights)  # 初始化权重

trainer = torch.optim.SGD(net.parameters(), lr=lr)  # 梯度下降优化器
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)  # 训练模型
