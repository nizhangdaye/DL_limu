import torch
from torch import nn
from d2l import torch as d2l


# 实现基本的转置卷积运算
# 定义转置卷积运算
def trans_conv(X, K):
    h, w = K.shape  # 卷积核的宽、高
    # 创建一个新的张量Y，其尺寸为输入X的尺寸加上卷积核K的尺寸减去1
    # 在常规卷积中，输出尺寸通常是输入尺寸减去卷积核尺寸加1
    Y = torch.zeros(
        (X.shape[0] + h - 1, X.shape[1] + w - 1))  # 正常的卷积后尺寸为(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # 对于输入X的每一个元素，我们将其与卷积核K进行元素级别的乘法，然后将结果加到输出张量Y的相应位置上
            Y[i:i + h, j:j + w] += X[i, j] * K  # 按元素乘法，加回到自己矩阵
    # 返回转置卷积的结果
    return Y


X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))
# tensor([[ 0.,  0.,  1.],
#         [ 0.,  4.,  6.],
#         [ 4., 12.,  9.]])

# 使用高级API获得相同的结果
# 将输入张量X和卷积核K进行形状变换，原来是2x2的二维张量，现在变成了1x1x2x2的四维张量
# 第一个1表示批量大小，第二个1表示通道数，2x2是卷积核的高和宽
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
# 创建一个转置卷积层对象tconv，其中输入通道数为1，输出通道数为1，卷积核的大小为2，没有偏置项
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
# 将创建的转置卷积层对象tconv的权重设置为我们的卷积核K
tconv.weight.data = K
print(tconv(X))
# tensor([[[[ 0.,  0.,  1.],
#           [ 0.,  4.,  6.],
#           [ 4., 12.,  9.]]]], grad_fn=<ConvolutionBackward0>)

# 填充、步幅和多通道

# 填充大小为1就相当于将输出矩阵最外面的一圈当作填充并剔除
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))
# tensor([[[[4.]]]], grad_fn=<ConvolutionBackward0>)

# 步幅为2表示在进行卷积时，每次移动2个单位，相较于步幅为1，这样会使得输出尺寸增大
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X))
# tensor([[[[0., 0., 0., 1.],
#           [0., 0., 2., 3.],
#           [0., 2., 0., 3.],
#           [4., 6., 6., 9.]]]], grad_fn=<ConvolutionBackward0>)

# 多通道
# 创建一个四维张量X，批量大小为1，通道数为10，高和宽都为16
X = torch.rand(size=(1, 10, 16, 16))
# 卷积层 将 10 个输入通道转换为 20 个输出通道
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
# 转置卷积层 将 20 个输入通道转换为 10 个输出通道
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
# 检查转置卷积后的输出尺寸是否与输入尺寸相同
print(tconv(conv(X)).shape == X.shape)
# True

# 与矩阵变换的联系
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)  # 卷积
print(Y)
# tensor([[27., 37.],
#         [57., 67.]])

def kernel2matrix(K):
    """
    用于将给定的卷积核K转换为一个稀疏矩阵W
    """
    k, W = torch.zeros(5), torch.zeros((4, 9))
    print(k)
    print(W)
    print(K)
    # 将卷积核K的元素填充到向量k中的适当位置，形成一个稀疏向量
    k[:2], k[3:5] = K[0, :], K[1, :]
    # 打印填充后的向量k
    print(k)
    # 将稀疏向量k填充到矩阵W中的适当位置，形成一个稀疏矩阵
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W


# 使用kernel2matrix函数将卷积核K转换为一个稀疏矩阵W
# 这个矩阵的每一行表示在一个特定位置进行的卷积操作，其中的0表示卷积核没有覆盖的区域
# 如果输入是一个3x3的图像，并被拉平为一个1x9的向量
# 而卷积核是2x2的，那么输出图像的大小为2x2，拉平后变为一个4x1的向量
# kernel2matrix函数实际上就是在构建这种转换关系
W = kernel2matrix(K)
print(W)
# tensor([0., 0., 0., 0., 0.])
# tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# tensor([[1., 2.],
#         [3., 4.]])
# tensor([1., 2., 0., 3., 4.])
# tensor([[1., 2., 0., 3., 4., 0., 0., 0., 0.],
#         [0., 1., 2., 0., 3., 4., 0., 0., 0.],
#         [0., 0., 0., 1., 2., 0., 3., 4., 0.],
#         [0., 0., 0., 0., 1., 2., 0., 3., 4.]])

print(X)
# tensor([[0., 1., 2.],
#         [3., 4., 5.],
#         [6., 7., 8.]])
print(X.reshape(-1))  # 拉平输入张量X
# tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.])

# 判断卷积操作的结果Y是否等于稀疏矩阵W与拉平的输入张量X的矩阵乘法的结果，并将结果重塑为2x2的形状
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))
# tensor([[True, True],
#         [True, True]])

Z = trans_conv(Y, K)
# 判断转置卷积操作的结果Z是否等于稀疏矩阵W的转置与拉平的卷积结果Y的矩阵乘法的结果，并将结果重塑为3x3的形状
# 注意这里得到的结果并不是原图像，尽管它们的尺寸是一样的
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))  # 由卷积后的图像乘以转置卷积后，得到的并不是原图像，而是尺寸一样
# tensor([[True, True, True],
#         [True, True, True],
#         [True, True, True]])
