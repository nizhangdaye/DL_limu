import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 生成数据集
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练集和测试集的大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  # 存储多项式的系数（权重）
# 随机生成 x
features = np.random.normal(size=(n_train + n_test, 1))  # 生成一个大小为(n_train + n_test, 1)的服从标准正态分布的随机特征数组
np.random.shuffle(features)  # 打乱特征数据的顺序
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))  # 对第所有维的特征取0次方、1次方、2次方...19次方
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # i次方的特征除以(i+1)阶乘
# 创建标签
labels = np.dot(poly_features, true_w)  # 根据多项式生成y，即生成真实的labels
labels += np.random.normal(scale=0.1, size=labels.shape)  # 为标签添加服从正态分布的噪声，以模拟真实数据的随机性

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32)
                                           for x in [true_w, features, poly_features, labels]]

print(features[:2], poly_features[:2, :], labels[:2], sep='\n')


# tensor([[-0.8282],
#         [ 1.4003]])
# tensor([[ 1.0000e+00, -8.2825e-01,  3.4300e-01, -9.4695e-02,  1.9608e-02,
#          -3.2480e-03,  4.4836e-04, -5.3050e-05,  5.4923e-06, -5.0545e-07,
#           4.1863e-08, -3.1521e-09,  2.1756e-10, -1.3861e-11,  8.2003e-13,
#          -4.5279e-14,  2.3439e-15, -1.1419e-16,  5.2545e-18, -2.2905e-19],
#         [ 1.0000e+00,  1.4003e+00,  9.8041e-01,  4.5762e-01,  1.6020e-01,
#           4.4866e-02,  1.0471e-02,  2.0946e-03,  3.6663e-04,  5.7044e-05,
#           7.9878e-06,  1.0168e-06,  1.1866e-07,  1.2781e-08,  1.2784e-09,
#           1.1934e-10,  1.0444e-11,  8.6030e-13,  6.6927e-14,  4.9325e-15]])
# tensor([2.4621, 5.9544])

# 评估模型
def evaluate_loss(net, data_iter, loss):  # @save
    """评估给定数据集上模型的损失。"""
    metric = d2l.Accumulator(2)  # 两个数的累加器
    for X, y in data_iter:
        out = net(X)  # 预测输出
        y = y.reshape(out.shape)  # 标签的形状需要和预测输出的形状一致
        l = loss(out, y)  # 计算损失
        metric.add(l.sum(), l.numel())  # 累加损失 和 样本数量
    return metric[0] / metric[1]  # 计算平均损失


# 训练函数
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')  # 均方误差损失
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))  # 定义模型
    batch_size = min(10, train_labels.shape[0])  # 训练集的批量大小
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)),
                                batch_size)  # 训练集数据迭代器
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)),
                               batch_size, is_train=False)  # 测试集数据迭代器
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)  # 小批量随机梯度下降
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])  # 绘制训练过程
    for epoch in range(num_epochs):  # 训练模型
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)  # 训练模型一个 epoch
        if epoch == 0 or (epoch + 1) % 20 == 0:  # 每 20 个 epoch 评估模型
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())


# 训练模型

# # 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
# train(poly_features[:n_train, :4], poly_features[n_train:, :4],
#       labels[:n_train], labels[n_train:])
# # weight: [[ 4.9890676  1.1630476 -3.4193513  5.6815   ]]

# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
# weight: [[2.9832623 4.2033267]]

# # 从多项式特征中选取所有维度
# train(poly_features[:n_train, :], poly_features[n_train:, :],
#       labels[:n_train], labels[n_train:], num_epochs=1500)
# # weight: [[ 4.9926720e+00  1.2720633e+00 -3.2974391e+00  5.2138767e+00
# #   -3.4428847e-01  1.3618331e+00  1.5626380e-01  1.1630570e-01
# #    1.5991014e-01  2.8070834e-02 -1.0929724e-01  2.7722307e-03
# #    7.9240397e-02  1.4077527e-02  5.4253254e-02  1.8295857e-01
# #    1.6828278e-01 -3.8895667e-02  1.5147164e-02  1.3848299e-01]]
