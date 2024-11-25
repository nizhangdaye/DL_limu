#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# In[ ]:


# 独热编码
# 打印词汇表的大小
print(len(vocab))
# 使用独热编码将 [0, 2] 表示的物体下标转换为独热向量，其中0表示第一个元素，2表示第3个元素
F.one_hot(torch.tensor([0, 2]), len(vocab))
# 28
# tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0]])


# In[ ]:


# 小批量形状是(批量大小，时间步数)
X = torch.arange(10).reshape((2, 5))
# 对X的转置进行独热编码，其中28表示编码长度，返回独热编码后的形状
F.one_hot(X.T, 28).shape


# torch.Size([5, 2, 28])


# In[ ]:


def get_params(vocab_size, num_hiddens, device):
    """
    初始化循环神经网络模型的模型参数
    :param vocab_size: 词汇表大小
    :param num_hiddens: 隐藏层大小
    :return: 返回模型参数（权重矩阵和偏置向量）
    """
    num_inputs = num_outputs = vocab_size  # 输入输出为经 one-hot 编码后的词汇向量

    def normal(shape):
        """生成指定形状的随机张量"""
        return torch.randn(size=shape, device=device) * 0.01

    # 初始化模型参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)  # 设置参数为需要梯度更新

    return params


# In[ ]:


def init_rnn_state(batch_size, num_hiddens, device):
    """初始化为零的隐藏状态"""
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# In[ ]:


def rnn(inputs, state, params):
    """
    定义循环神经网络模型，包括了所有时间步的计算，类似于前向传播
    :param inputs: 输入序列形状为 (时间步数，批量大小, 词汇表大小)
    :param state: 隐藏状态，形状为 (批量大小, 隐藏层大小)
    :param params: 模型参数，包含权重矩阵和偏置向量
    :return: 返回输出序列和新的隐藏状态，Y 形状为 (时间步数 * 批量大小, 输出大小)
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:  # 对于每个时间步
        # 根据当前输入 X、上一时间步的隐藏状态 H、以及权重矩阵和偏置向量来计算
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)  # 更新隐状态
        # 计算输出 Y，通过隐藏状态 H 与权重矩阵 W_hq 相乘并加上偏置向量 b_q 得到
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)  # 形状：(时间步数 * 批量大小, 输出大小)


# In[ ]:


# 创建一个类来包装这些函数
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        """初始化模型"""
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens  # 词汇表大小，隐藏层大小
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn  # 隐藏状态初始化函数，前向传播函数（rnn）

    def __call__(self, X, state):
        # 将输入序列 X 进行独热编码，形状为 (时间步数, 批量大小, 词汇表大小)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # 调用前向传播函数进行模型计算，并返回输出，形状为 (时间步数 * 批量大小, 输出大小)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        # 返回初始化的隐藏状态，用于模型的初始时间步
        return self.init_state(batch_size, self.num_hiddens, device)


# In[ ]:


# 检查输出是否具有正确的形状
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())  # （批量大小，隐藏层大小）
Y, new_state = net(X.to(d2l.try_gpu()), state)  # 前向传播
Y.shape, type(new_state), len(new_state), new_state[0].shape


# (torch.Size([10, 28]), tuple, 1, torch.Size([2, 512]))


# In[ ]:


def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符
    :param prefix: 字符串，表示要生成新字符的前缀
    :param num_preds: 要生成的新字符的数量
    :param net: 循环神经网络模型
    :param vocab: 词汇表
    """
    state = net.begin_state(batch_size=1, device=device)  # 隐藏状态初始化
    outputs = [vocab[prefix[0]]]  # 第一个输出，方便 get_input 函数获取
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(
        (1, 1))  # 张量化函数，将最近预测的字符作为输入，形状为 (时间步数, 批量大小)
    for y in prefix[1:]:  # 预热期间的输入，此时在更新隐藏状态
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())


# 'time traveller lkxdcccccc'


# In[ ]:


def grad_clipping(net, theta):
    """
    裁剪梯度
    :param net: 循环神经网络模型
    :param theta: 裁剪阈值
    """
    if isinstance(net, nn.Module):  # 如果 net 是 nn.Module 的实例
        # 获取所有需要计算梯度的参数列表
        params = [p for p in net.parameters() if p.requires_grad]
    else:  # 如果 net 是自定义的模型
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))  # 计算梯度范数
    if norm > theta:  # 如果梯度范数大于裁剪阈值
        for param in params:
            param.grad[:] *= theta / norm  # 缩放梯度


# In[ ]:


# 定义一个函数来训练只有一个迭代周期的模型
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """
    训练模型一个迭代周期
    :param net: 循环神经网络模型
    :param train_iter: 训练数据集
    :param loss: 损失函数
    :param updater: 自定义的更新函数或 PyTorch 内置的优化器
    :param use_random_iter: 是否使用随机迭代器
    """
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 初始化度量指标的累加器，用于计算损失和样本数量
    for X, Y in train_iter:
        if state is None or use_random_iter:  # 如果隐藏状态为空或使用随机迭代器
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:  # 使用上一时间步的隐藏状态
            if isinstance(net, nn.Module) and not isinstance(state, tuple):  # 如果 net 是 nn.Module 的实例且 state 不是 tuple
                state.detach_()  # 对于 nn.Module 的实例，分离隐藏状态的计算图，用于减少内存占用和加速计算
            else:  # 如果 net 是自定义的模型或 state 是 tuple
                for s in state:
                    s.detach_()  # 对于自定义的模型或 state 是 tuple，分离隐藏状态的计算图
        y = Y.T.reshape(-1)  # 转置后，形状为 (时间步数, 批量大小,)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)  # 前向传播计算预测值和新的隐藏状态
        # 一下为标准的多分类问题
        l = loss(y_hat, y.long()).mean()  # 计算交叉熵损失
        if isinstance(updater, torch.optim.Optimizer):  # 如果 updater 是 torch 内置的优化器
            updater.zero_grad()  # 梯度清零
            l.backward()  # 反向传播计算梯度
            grad_clipping(net, 1)  # 裁剪梯度
            updater.step()  # 使用优化器更新参数
        else:
            l.backward()  # 反向传播计算梯度
            grad_clipping(net, 1)  # 裁剪梯度
            updater(batch_size=1)  # 更新参数
        metric.add(l * y.numel(), y.numel())  # 累加损失和样本数量
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# In[ ]:


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """
    训练模型
    :param net: 循环神经网络模型
    :param train_iter: 训练数据集
    :param vocab: 词汇表
    :param use_random_iter: 是否使用随机迭代器
    """
    loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    if isinstance(net, nn.Module):  # 如果 net 是 nn.Module 的实例
        updater = torch.optim.SGD(net.parameters(), lr)  # 使用 SGD 优化器
    else:  # 否则，使用自定义的梯度下降函数进行参数更新
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)  # 定义预测函数
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 标记/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


# In[ ]:


# 现在我们可以训练循环神经网络模型
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())  # 循序分区

# In[ ]:


# 最后，让我们检查一下使用随即抽样方法的结果
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), use_random_iter=True)  # 随机抽样分区
