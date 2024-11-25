import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

tau = 4
features = torch.zeros((T - tau, tau))
# 每四个数据作为特征，第五个作为标签，不断构造这样的数据形成数据集
for i in range(tau):
    features[:, i] = x[i:T - tau + i]  # 前4个时刻的数值组成一个向量作为特征
# 所从第5个时刻开始，每个时刻的label是该时刻的x值，该时刻的输入是前4个时刻的数值组成的一个向量。
# 经过变化后数据的输入共有996组4个一组的数据，输出共996个值
labels = x[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600  # 取前600个样本作为训练集
# 使用 features 和 labels 的前 n_train 个样本创建一个可迭代的训练集
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


def init_weights(m):
    # 如果当前模块是线性层
    if type(m) == nn.Linear:
        # 初始化权重函数
        nn.init.xavier_uniform_(m.weight)


def get_net():
    # 定义神经网络结构
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    # 对网络的权重进行初始化
    net.apply(init_weights)
    # 返回构建好的神经网络模型
    return net


# 定义均方误差损失函数
loss = nn.MSELoss()


# 定义训练函数
def train(net, train_iter, loss, epochs, lr):
    # 定义优化器
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()  # 梯度清零
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


# 训练模型
net = get_net()
train(net, train_iter, loss, 5, 0.01)
# epoch 1, loss: 0.067933
# epoch 2, loss: 0.056947
# epoch 3, loss: 0.053831
# epoch 4, loss: 0.055126
# epoch 5, loss: 0.056145

# 预测模型
onestep_preds = net(features)
# 进行数据可视化，将真实数据和一步预测结果绘制在同一个图中进行比较
d2l.plt.figure()
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time', 'x',
         legend=['data', 'l-step preds'], xlim=[1, 1000], figsize=(6, 3))

# 初始化多步预测结果的张量
multistep_preds = torch.zeros(T)
# 将已知的真实数据赋值给多步预测结果
multistep_preds[:n_train + tau] = x[:n_train + tau]
# 对剩余时间步进行多步预测
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))  # 前4个时刻的数值预测第5个时刻的数值

# 进行数据可视化
d2l.plt.figure()
d2l.plot(
    [time, time[tau:], time[n_train + tau:]],
    [x.detach().numpy(), onestep_preds.detach().numpy(), multistep_preds[n_train + tau:].detach().numpy()],
    'time',
    'x',
    legend=['data', '1-step preds', 'multistep preds'],
    xlim=[1, 1000],
    figsize=(6, 3))

max_steps = 64  # 最多预测未来的64步
# 初始化特征张量
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
for i in range(tau):
    # 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
    features[:, i] = x[i:i + T - tau - max_steps + 1]

# 预测未来max_steps步
for i in range(tau, tau + max_steps):
    # 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)  # 根据前4个时刻的数值（包括已经预测的部分）预测第5个时刻的数值

# 预测的步长
steps = (1, 4, 16, 64)  # 第1步，第4步，第16步，第64步的预测结果
# 进行数据可视化
d2l.plt.figure()
d2l.plot([time[tau + i - 1:T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps],
         'time',
         'x',
         legend=[f'{i}-step preds' for i in steps],
         xlim=[5, 1000],
         figsize=(6, 3))
