import torch
from torch import nn
from d2l import torch as d2l

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义 L_2 范数惩罚项
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


# 定义训练函数
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss  # 线性回归模型和平方损失函数
    num_epochs, lr = 100, 0.003  # 训练周期和学习率
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:  # 每5个周期评估模型
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())


# 忽略L2范数惩罚项的训练结果
train(lambd=0)
# w的L2范数是： 14.708277702331543
# 加入L2范数惩罚项的训练结果
train(lambd=3)
# w的L2范数是： 0.36321237683296204
train(lambd=10)
# w的L2范数是： 0.019130496308207512
train(lambd=100)


# w的L2范数是： 0.004614777863025665

# 从上面的实验结果可以看出，L2范数惩罚项可以使模型参数更加稀疏，从而减少过拟合。
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))  # 线性回归模型 y = Xw + b
    for param in net.parameters():  # 初始化模型参数
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')  # 均方误差损失函数 reduction='none'表示返回每个样本的损失
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([  # 随机梯度下降优化器
        {"params": net[0].weight, 'weight_decay': wd},  # 权重参数衰减
        {"params": net[0].bias}], lr=lr)  # 偏置参数不衰减
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()  # 梯度清零
            l = loss(net(X), y)  # 计算损失
            l.mean().backward()  # 梯度反向传播
            trainer.step()  # 更新参数
        if (epoch + 1) % 5 == 0:  # 每5个周期评估模型
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())


train_concise(0)
train_concise(3)
train_concise(100)
