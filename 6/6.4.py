import torch
from d2l import torch as d2l
import time

start = time.perf_counter()


# 多输入通道
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))  # 这里的zip函数可以把X和K分别取出一个元素，然后把它们作为参数传入corr2d函数


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]],
                  [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X, K))
# tensor([[ 56.,  72.],
#         [104., 120.]])


# 多输出通道
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)  # stack函数将结果堆叠起来


K = torch.stack([K, K + 1, K + 2], 0)  # 3个通道的卷积核
print(K.shape)
# torch.Size([3, 2, 2, 2])
print(corr2d_multi_in_out(X, K))
# tensor([[[ 56.,  72.],
#          [104., 120.]],
#         [[ 76., 100.],
#          [148., 172.]],
#         [[ 96., 128.],
#          [192., 224.]]])

# 1×1卷积的多输入、多输出通道运算
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape  # 输入的通道数、宽、高
    c_o = K.shape[0]  # 输出的通道数
    X = X.reshape((c_i, h * w))  # 拉平操作，每一行表示一个通道的特征
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


X = torch.normal(0, 1, (3, 3, 3))  # norm函数生成0到1之间的(3,3,3)矩阵
K = torch.normal(0, 1, (2, 3, 1, 1))  # 输出通道是2，输入通道是3，核是1X1
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
print(float(torch.abs(Y1 - Y2).sum()))
# 0.0


end = time.perf_counter()
print("运行耗时", end - start)
