```toc
```

- 之前讨论的多层感知机十分适合处理表格数据，其中行对应样本，列对应特征
    - 对于表格数据，我们寻找的模式可能涉及特征之间的交互，但是我们不能预先假设任何与特征交互相关的先验结构
    - 然而对于高维感知数据，这种缺少结构的网络可能会变得不实用
- 假设我们有一个足够充分的照片数据集，数据集中是拥有标注的照片，每张照片具有百万级像素，这意味着网络的每次输入都有一百万个维度。
  即使将隐藏层维度降低到1000，这个全连接层也将有 $10^6×10^3=10^9$ 个参数
    - 而且拟合如此多的参数还需要收集大量的数据
- 使用 MLP 十分消耗资源![[00 Attachments/Pasted image 20240530213956.png|400]]

## 从全连接到卷积

- 对全连接层使用平移不变性和局部性得到卷积层

### 平移不变性与局部性

![[00 Attachments/Pasted image 20240530214446.png|400]]

- 平移不变性：如果要找一张图片中的物体：无论哪种方法找到这个物体，都应该和物体的位置无关
- 局部性：可以使用一个“检测器”扫描图像。 该检测器将图像分割成多个区域，并为每个区域包含目标的可能性打分。
  卷积神经网络正是将空间不变性（spatial invariance）的这一概念系统化，从而基于这个模型使用较少的参数来学习有用的表示
    - 不需要再全图找
- 从而设计合适的计算机视觉神经网络架构
    1. 平移不变性（translation invariance）：不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。
    2. 局部性（locality）：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。

### 多层感知机的限制

- 多层感知机的输入为一个二维图像 $\mathbf{X}$。隐藏表示 $\mathbf{H}$ 在数学上是一个矩阵，在代码中表示为二维张量
    - 输入和隐藏都具有相同的空间结构
- 使用 $[\mathbf{X}]_{i, j}$ 和 $[\mathbf{H}]_{i, j}$ 分别表示输入图像和隐藏表示中位置（𝑖,𝑗）处的像素
- $$\begin{split}\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
  \sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}\end{split}$$
    - 从 𝑊 到 𝑉 的转换只是形式上的转换，因为在这两个四阶张量的元素之间存在一一对应的关系
    - 只需重新索引下标 $(𝑘,𝑙)$，使 $𝑘=𝑖+𝑎、𝑙=𝑗+𝑏$，由此可得 $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$。 索引
      𝑎 和 𝑏 通过在正偏移和负偏移之间移动覆盖了整个图像
    - 对于隐藏表示中任意给定位置 $(𝑖,𝑗)$ 处的像素值 $[\mathbf{H}]_{i, j}$，可以通过在 𝑥 中以 $(𝑖,𝑗)$
      为中心对像素进行加权求和得到，加权使用的权重为 $[\mathsf{V}]_{i, j, a, b}$
    - $\mathbf{V}$ 被称为卷积核（convolution kernel）或者滤波器（filter），亦或简单地称之为该卷积层的_权重_，通常该权重是可学习的参数
- ![[00 Attachments/Pasted image 20240530223234.png|400]]从而得到![[00 Attachments/Pasted image 20240530224435.png|400]]

### 平移不变性

- 当在图片中形成一个识别器后，在一定像素大小的范围内，它都有自己的权重，当这个识别器在图片上换位置之后，它的权重应该不变。
- 理解成用同一张卷积核遍历整张图片。卷积核不会随着位置变化而变化。
- 权重就是特征提取器，不应该随位置而发生变化。
- 简而言之卷积核就是个框，在图片上不断扫描，无论扫在图上的哪个位置，卷积核都是不变的。
- 对于一张图片应该有多个卷积核，但是每个卷积核要识别的东西不同，一个卷积核就是一个分类器。
- 卷积确实是 weight shared，但不是全联接，每个神经元是对应卷积核大小个输入
- 卷积就是 weight shared 全连接。
- ![[00 Attachments/Pasted image 20240530223427.png|400]]

### 局部性

- 指取一个不那么大的框（指卷积核的大小）
  ![[00 Attachments/Pasted image 20240530223712.png|400]]

## 卷积层

- 卷积层的输入和输出由四维张量组成，张量的每个轴分别对应样本、通道、高度和宽度

### 图像卷积

- ![[00 Attachments/Pasted image 20240530224632.png|400]]
- 输出相比于输入都会缩小![[00 Attachments/Pasted image 20240530225647.png|400]]
- 举例![[00 Attachments/Pasted image 20240530225739.png|400]]
- ![[00 Attachments/Pasted image 20240530230153.png|400]]
- ![[00 Attachments/Pasted image 20240530230202.png|400]]
- 总结![[00 Attachments/Pasted image 20240530230310.png|400]]

#### 相关代码

##### 互相关运算

- 该函数接受输入张量`X`和卷积核张量`K`，并返回输出张量`Y`

```python
import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):  # X 为输入，K为核矩阵  
    """计算二维互相关信息"""
    h, w = K.shape  # 核矩阵的行数和列数  
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))  # 输出图像的形状 由公式计算得出  
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()  # 图片的小方块区域与卷积核做点积  
    return Y


# 验证上述二维互相关运算的输出  
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))
# tensor([[19., 25.],  
#         [37., 43.]])
```

##### 二维卷积层

- 卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出。 所以，==卷积层中的两个被训练的参数是卷积核权重和标量偏置==。
    - 就像我们之前随机初始化全连接层一样，在训练基于卷积层的模型时，我们也随机初始化卷积核权重
- 基于上面定义的`corr2d`函数实现二维卷积层。在`__init__`构造函数中，将`weight`和`bias`声明为两个模型参数。前向传播函数调用
  `corr2d`函数并添加偏置

```python
# 卷积层的一个简单应用：检测图片中不同颜色的边缘  
X = torch.ones((6, 8))
X[:, 2:6] = 0  # 把中间四列设置为0  
print(X)  # 0 与 1 之间进行过渡，表示边缘  

K = torch.tensor([[1.0, -1.0]])  # 如果左右原值相等，那么这两原值乘1和-1相加为0，则不是边缘  高度为1、宽度为2的卷积核K
Y = corr2d(X, K)
print(Y)
print(corr2d(X.t(), K))  # X.t() 为X的转置，而K卷积核只能检测垂直边缘  
# tensor([[1., 1., 0., 0., 0., 0., 1., 1.],  
#         [1., 1., 0., 0., 0., 0., 1., 1.],  
#         [1., 1., 0., 0., 0., 0., 1., 1.],  
#         [1., 1., 0., 0., 0., 0., 1., 1.],  
#         [1., 1., 0., 0., 0., 0., 1., 1.],  
#         [1., 1., 0., 0., 0., 0., 1., 1.]])  
# tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],  
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],  
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],  
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],  
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],  
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])  
# tensor([[0., 0., 0., 0., 0.],  
#         [0., 0., 0., 0., 0.],  
#         [0., 0., 0., 0., 0.],  
#         [0., 0., 0., 0., 0.],  
#         [0., 0., 0., 0., 0.],  
#         [0., 0., 0., 0., 0.],  
#         [0., 0., 0., 0., 0.],  
#         [0., 0., 0., 0., 0.]])
```

- 输出`Y`中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为0

##### 由 X 生成 Y 的卷积核

- 上述卷积核是自定义的，但一旦涉及更加复杂的设计，就不能手动设计卷积核
- 通过仅查看“输入-输出”对来学习由`X`生成`Y`的卷积核
    - 先构造一个卷积层，并将其卷积核初始化为随机张量
    - 在每次迭代中，我们比较`Y`与卷积层输出的平方误差，然后计算梯度来更新卷积核
- 为了简单起见，我们在此使用内置的二维卷积层，并忽略偏置

```python
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核  
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），  
# 其中批量大小和通道数都为1  
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率  

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2  # 损失函数  
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核  
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
    # epoch 2, loss 8.822  
# epoch 4, loss 2.950  
# epoch 6, loss 1.097  
# epoch 8, loss 0.431  
# epoch 10, loss 0.173
```

- 在10次迭代之后，误差已经降到足够低。现在我们来看看我们所学的卷积核的权重张量

```python
print(conv2d.weight.data.reshape((1, 2)))
# tensor([[ 0.9487, -1.0341]])
```

- 可以发现非常接近之前定义的卷积核

### 卷积层里的填充和步幅

- ![[00 Attachments/Pasted image 20240530230749.png|400]]
    - 每过一层，图像大小减去 4

#### 填充（padding）

- 可以看出通过卷积只能做到 7 层，但是深度学习是如何使用更深的模型（几百层）
    - 希望在卷积核大小不变的情况下，使用更多的层数
- ![[00 Attachments/Pasted image 20240530231207.png|400]]
- ![[00 Attachments/Pasted image 20240530231653.png|400]]
    - 在许多情况下，我们需要设置 $𝑝_ℎ=𝑘_ℎ−1$ 和 $𝑝_𝑤=𝑘_𝑤−1$，使输入和输出具有相同的高度和宽度。 这样可以在构建网络时更容易地预测每个图层的输出形状
        - 假设 $𝑘_ℎ$ 是奇数，我们将在高度的两侧填充 $𝑝_ℎ/2$ 行。 如果 $𝑘_ℎ$ 是偶数，则一种可能性是在输入顶部填充 $⌈𝑝_ℎ/2⌉$
          行，在底部填充 $⌊𝑝_ℎ/2⌋$ 行。同理，我们填充宽度的两侧
    - 奇数卷积核更容易做 padding。我们假设卷积核大小为 k *
      k，为了让卷积后的图像大小与原图一样大，根据公式可得到 $padding=(k-1)/2$，这里的k只有在取奇数的时候，padding 才能是整数，否则
      padding 不好进行图片填充
- 此外，使用奇数的核大小和填充大小也提供了书写上的便利。对于任何二维张量`X`，当满足：
    1. 卷积核的大小是奇数
    2. 所有边的填充行数和列数相同
    3. 输出与输入具有相同高度和宽度
- 则可以得出：输出`Y[i, j]`是通过以输入`X[i, j]`为中心，与卷积核进行互相关计算得到的

##### 代码

- 创建一个高度和宽度为3的二维卷积层，并在所有侧边填充1个像素。给定高度和宽度为8的输入，则输出的高度和宽度也是8

```python
import torch
from torch import nn


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

- 当卷积核的高度和宽度不同时，我们可以填充不同的高度和宽度，使输出和输入具有相同的高度和宽度
- 使用高度为5，宽度为3的卷积核，高度和宽度两边的填充分别为2和1

```python
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)
# torch.Size([8, 8])
```

#### 步幅

- 如果想使用较小的层数以更快的得到输出
- ![[00 Attachments/Pasted image 20240530232152.png|400]]
- ![[00 Attachments/Pasted image 20240530232335.png|400]]
- ![[00 Attachments/Pasted image 20240530232346.png|400]]

##### 代码

- 将高度和宽度的步幅设置为2，从而将输入的高度和宽度减半

```python
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)
# torch.Size([4, 4])
```

-

```python
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)
# torch.Size([2, 2])
```

- 可以套公式得出

#### 总结

- ![[00 Attachments/Pasted image 20240530232754.png|400]]

### 卷积层里的多输入多输出通道

#### 多输入通道

-
输入不再是一个矩阵，而是一个有三个通道的彩图![[00 Attachments/Pasted image 20240530233751.png|400]]![[00 Attachments/Pasted image 20240530233759.png|400]]
- 如果图片是 200\*200 那么张量的表示因该是 200\*200\*3
- ![[00 Attachments/Pasted image 20240530233953.png|400]]
- 核的通道数与输入的通道数一样![[00 Attachments/Pasted image 20240530234242.png|300]]
    - 对于每一个多通道输入，将其各个通道卷积对应的卷积核之后相加
- 简而言之，我们所做的就是对每个通道执行互相关操作，然后将结果相加

```python
import torch
from d2l import torch as d2l


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
```

#### 多输出通道

- 到目前为止，不论有多少输入通道，我们还只有一个输出通道
- 最流行的神经网络架构中，随着神经网络层数的加深，我们常会增加输出通道的维数，通过减少空间分辨率以获得更大的通道深度
    - 直观地说，我们可以将每个通道看作对不同特征的响应（抽取不同特征）
    - 而现实可能更为复杂一些，因为每个通道不是独立学习的，而是为了共同使用而优化的
        - 因此，多输出通道并不仅是学习多个单通道的检测器
- ![[00 Attachments/Pasted image 20240530235210.png|400]]
- 一个核负责找一个特征，输出该特征代表的特征图![[00 Attachments/Pasted image 20240531003908.png|400]]
    - 寻找这六个特征
    - 最后可以将这些特征按重要性（权重）相加得到一个组合的模式（特征）识别
        - 猫眼睛 + 猫耳朵 -> 猫头
- 实现一个计算多个通道的输出的互相关函数

```python
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。  
    # 最后将所有结果都叠加在一起  
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

- 通过将核张量`K`与`K+1`（`K`中每个元素加1）和`K+2`连接起来，构造了一个具有3个输出通道的卷积核

```python
K = torch.stack([K, K + 1, K + 2], 0)  # 3个输出通道的卷积核  
print(K.shape)
# torch.Size([3, 2, 2, 2])
```

- 两个通道的输入，通过一个四维的卷积核（输出通道、输入通道、行、列），得到三个通道的输出

```python
print(corr2d_multi_in_out(X, K))
# tensor([[[ 56.,  72.],  
#          [104., 120.]],  
#         [[ 76., 100.],  
#          [148., 172.]],  
#         [[ 96., 128.],  
#          [192., 224.]]])
```

#### 1×1 卷积层

- ==等价于一个全连接层==（为什么？）
- 因为使用了最小窗口，1×1 卷积失去了卷积层的特有能力——在高度和宽度维度上，识别相邻元素间相互作用的能力（空间信息）
- 对多个通道的相同位置的像素进行融合（压缩饼干）![[00 Attachments/Pasted image 20240531004936.png|400]]
    - 这里输入和输出具有相同的高度和宽度，输出中的每个元素都是从输入图像中同一位置的元素的线性组合
    - 我们可以将1×1卷积层看作在每个像素位置应用的全连接层，以 $𝑐_𝑖$ 个输入值转换为 $𝑐_𝑜$ 个输出值
        - 因为这仍然是一个卷积层，所以跨像素的权重是一致的
    - 同时，1×1卷积层需要的权重维度为𝑐𝑜×𝑐𝑖，再额外加上一个偏置
- 使用全连接层实现 $1×1$ 卷积

```python
# 1×1卷积的多输入、多输出通道运算
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape  # 输入的通道数、宽、高  
    c_o = K.shape[0]  # 输出的通道数  
    X = X.reshape((c_i, h * w))  # 拉平操作，每一行表示一个通道的特征  
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
```

- 当执行1×1卷积运算时，上述函数相当于先前实现的互相关函数`corr2d_multi_in_out`

```python
X = torch.normal(0, 1, (3, 3, 3))  # norm函数生成0到1之间的(3,3,3)矩阵  
K = torch.normal(0, 1, (2, 3, 1, 1))  # 输出通道是2，输入通道是3，核是1X1  
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
print(float(torch.abs(Y1 - Y2).sum()))
# 0.0
```

#### 二维卷积层

- 完整二维卷积层的所有参数![[00 Attachments/Pasted image 20240531005449.png|400]]
    - 一个卷积核对应一个偏差
    - 1G = 10亿 次浮点运算
    - 1 M(兆) = 100 万
    - 模型不大，但计算量大，扫 100遍 就要花 2h

#### 总结

- ![[00 Attachments/Pasted image 20240531010943.png|400]]
    - 输入通道数是前一层的超参数

## 池化层

- 实际图像里，我们感兴趣的物体不会总出现在固定像素位置：即使我们用三脚架固定相机去连续拍摄同一个物体也极有可能出现像素位置上的偏移
- 另外，绝大多数计算机视觉任务对图像处理终极目标是识别图片内的物体，所以不需要细致到对每个像素进行检测，只需要找到图片中物体的大概轮廓就好了
- ==池化层可以缓解卷积层对位置（细节）的过度敏感性==
- 如下所示，例如 $1×2$ 的卷积核 [1,-1]，会使得下图中 Y 输出的第二列为 1，其他为 0，如果像素偏移，会导致边缘检测的 1
  在其他位置输出，所以说卷积对像素的位置是非常敏感的![[00 Attachments/Pasted image 20240606201852.png|400]]
- 这里本质上讲的是池化层对于像素偏移的容忍性

### 最大池化层

-
先通过卷积，再通过池化![[00 Attachments/Pasted image 20240606202236.png|400]]![[00 Attachments/Pasted image 20240606202246.png|400]]

### 平均池化层

- 平均池化层相对最大池化层有柔和的效果![[00 Attachments/Pasted image 20240606202418.png|400]]

### 步长、步幅和多个通道

- ![[00 Attachments/Pasted image 20240606202742.png|400]]

### 总结

- 池化层返回窗口中最大或平均值
- 缓解卷积层队位置的敏感性
- 池化层的输出通道数与输入通道数相同
- 同样有窗口大小、填充和步幅作为超参数

### 代码实现

#### 最大池化层和平均池化层

- 没有卷积核，输出为输入中每个区域的最大值或平均值

```python
import torch
from torch import nn
from d2l import torch as d2l
import time

start = time.perf_counter()


# 实现池化层的正向传播  
def pool2d(X, pool_size, mode='max'):  # 拿到输入，池化窗口大小  
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))  # 输入的高减去窗口的高，再加上1，这里没有padding  
    for i in range(Y.shape[0]):  # 行遍历  
        for j in range(Y.shape[1]):  # 列遍历  
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y
```

- 验证二维最大汇聚层的输出

```python
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
# tensor([[4., 5.],  
#         [7., 8.]])
```

- 验证二维平均汇聚层的输出

```python
print(pool2d(X, (2, 2), 'avg'))
# tensor([[2., 3.],  
#         [5., 6.]])
```

#### 填充和步幅

- 与卷积层一样，汇聚层也可以通过填充和步幅改变输出形状
- 使用内置的二维最大池化层来演示填充和步幅
- 首先构造一个具有四个维度的输入张量 $X$

```python
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))  # 样本数，通道数，高，宽  
print(X)
# tensor([[[[ 0.,  1.,  2.,  3.],  
#          [ 4.,  5.,  6.,  7.],  
#          [ 8.,  9., 10., 11.],  
#          [12., 13., 14., 15.]]]])
```

- 默认情况下，深度学习框架中的步幅与池化窗口的大小相同
    - 因此，如果我们使用形状为`(3, 3)`的池化窗口，那么默认情况下，我们得到的步幅形状为`(3, 3)`

```python
pool2d = nn.MaxPool2d(3)  # 深度学习框架中的步幅默认与池化窗口的大小相同，下一个窗口和前一个窗口没有重叠的  
print(pool2d(X))
# tensor([[[[10.]]]])
```

- 也可以手动设定填充和步幅

```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
# tensor([[[[ 5.,  7.],  
#           [13., 15.]]]])
```

- 设计一个任意大小的矩形池化窗口，并分别设定填充和步幅的高度和宽度

```python
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))
# tensor([[[[ 5.,  7.],  
#           [13., 15.]]]])
```

#### 多个通道

- 在处理多通道输入数据时，==池化层在每个输入通道上单独运算==，而不是像卷积层一样在通道上对输入进行汇总
    - 这意味着池化层的输出通道数与输入通道数相同
- 在通道维度上连结张量`X`和`X + 1`，以构建具有2个通道的输入

```python
X = torch.cat((X, X + 1), 1)  # 沿指定维度合并两个样本  
print(X.shape)  # 合并起来，变成了1X2X4X4的矩阵  
print(X)
# tensor([[[[ 0.,  1.,  2.,  3.],  
#           [ 4.,  5.,  6.,  7.],  
#           [ 8.,  9., 10., 11.],  
#           [12., 13., 14., 15.]],  
#  
#          [[ 1.,  2.,  3.,  4.],  
#           [ 5.,  6.,  7.,  8.],  
#           [ 9., 10., 11., 12.],  
#           [13., 14., 15., 16.]]]])

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
# tensor([[[[ 5.,  7.],  
#           [13., 15.]],  
#
#          [[ 6.,  8.],  
#           [14., 16.]]]])
```

## 卷积神经网络（LeNet）

- ![[00 Attachments/Pasted image 20240610220252.png|400]]
    - 每个卷积块中的基本单元是一个卷积层、一个 sigmoid 激活函数和平均汇聚层
    - 每个卷积层使用 5×5 卷积核和一个 sigmoid 激活函数
        - 这些层将输入映射到多个二维特征输出，通常同时增加通道的数量
        - 第一卷积层有6个输出通道，而第二个卷积层有16个输出通道
    - 每个 2×2 池操作（步幅2）通过空间下采样将维数减少4倍。卷积的输出形状由批量大小、通道数、高度、宽度决定
        - 输出后的大小可由公式计算得到（卷积层的步幅里）
    - 为了将卷积块的输出传递给稠密块，我们必须在小批量中展平每个样本
- 示意图![[00 Attachments/Pasted image 20240610222245.png]]
- ![[00 Attachments/Pasted image 20240610220310.png|400]]

### 代码实现

#### LeNet

- 实例化一个 Sequential 块并将需要的层连接在一起
- ![[00 Attachments/Pasted image 20240610222245.png]]

```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

- 将一个大小为 28×28 的单通道（黑白）图像通过 LeNet。通过在每一层打印输出的形状，我们可以检查模型

```python
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
# Conv2d output shape:   torch.Size([1, 6, 28, 28])  
# Sigmoid output shape:      torch.Size([1, 6, 28, 28])  
# AvgPool2d output shape:    torch.Size([1, 6, 14, 14])  
# Conv2d output shape:   torch.Size([1, 16, 10, 10])  
# Sigmoid output shape:      torch.Size([1, 16, 10, 10])  
# AvgPool2d output shape:    torch.Size([1, 16, 5, 5])  
# Flatten output shape:      torch.Size([1, 400])  
# Linear output shape:   torch.Size([1, 120])  
# Sigmoid output shape:      torch.Size([1, 120])  
# Linear output shape:   torch.Size([1, 84])  
# Sigmoid output shape:      torch.Size([1, 84])  
# Linear output shape:   torch.Size([1, 10])
```

- 在整个卷积块中，与上一层相比，每一层特征的高度和宽度都减小了
    - 第一个卷积层使用 2 个像素的填充，来补偿 5×5 卷积核导致的特征减少。
    - 相反，第二个卷积层没有填充，因此高度和宽度都减少了4个像素。
    - 随着层叠的上升，通道的数量从输入时的1个，增加到第一个卷积层之后的6个，再到第二个卷积层之后的16个
    - 同时，每个汇聚层的高度和宽度都减半
    - 最后，每个全连接层减少维数，最终输出一个维数与结果分类数相匹配的输出

#### 模型训练

- 使用 Fashion-MNIST 数据集

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

-
为了进行评估，我们需要对[3.6节](https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression-scratch.html#sec-softmax-scratch)
中描述的`evaluate_accuracy`函数进行轻微的修改。
- 由于完整的数据集位于内存中，因此在模型使用GPU计算数据集之前，我们需要将其复制到显存中

```python
# 对evaluate_accuracy函数进行轻微的修改  
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # net.eval()开启验证模式，不用计算梯度和更新梯度  
        if not device:
            device = next(iter(net.parameters())).device  # 看net.parameters()中第一个元素的device为哪里  
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]  # 如果X是个List，则把每个元素都移到device上  
        else:
            X = X.to(device)  # 如果X是一个Tensor，则只用移动一次，直接把X移动到device上  
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())  # y.numel() 为y元素个数   


return metric[0] / metric[1]
```

- 为了使用GPU，我们还需要一点小改动
    - 与[3.6节](https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression-scratch.html#sec-softmax-scratch)中定义的
      `train_epoch_ch3`不同，在进行正向和反向传播之前，我们需要将每一小批量数据移动到我们指定的设备（例如GPU）上
- 由于我们将实现多层神经网络，因此我们将主要使用高级API
- 以下训练函数假定从高级API创建的模型作为输入，并进行相应的优化
    -
    我们使用在[4.8.2.2节](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#subsec-xavier)
    中介绍的Xavier随机初始化模型参数
    - 与全连接层一样，我们使用交叉熵损失函数和小批量随机梯度下降

```python
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""

    # 初始化模型参数  
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)  # 使用Xavier初始化权重  

    net.apply(init_weights)
    print('training on', device)
    net.to(device)  # 将模型参数移到device上  
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 小批量随机梯度下降  
    loss = nn.CrossEntropyLoss()  # 交叉熵损失函数  
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)  # 训练速度计时器  
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数  
        metric = d2l.Accumulator(3)
        net.train()  # 切换到训练模式  
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 梯度清零  
            X, y = X.to(device), y.to(device)  # 移到device上  
            y_hat = net(X)  # 前向传播计算预测值  
            l = loss(y_hat, y)  # 计算损失  
            l.backward()
            optimizer.step()  # 优化器更新参数  
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:  # 每5个batch输出一次信息  
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 在测试集上评价模型  
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')  # 输出训练结果  
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')  # 输出训练速度
```

- 进行训练

```python
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# training on cuda:0  
# loss 0.473, train acc 0.820, test acc 0.801  
# 27967.0 examples/sec on cuda:0
```

- ![[00 Attachments/Figure_11.png]]
