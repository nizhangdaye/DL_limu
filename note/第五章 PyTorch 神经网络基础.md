```toc
```

## 层和块

- 当考虑具有多个输出的网络时， 我们利用矢量化算法来描述整层神经元。 像单个神经元一样，层
    - （1）接受一组输入
    - （2）生成相应的输出
    - （3）由一组可调整参数描述
- 为了实现这些复杂的网络，我们引入了神经网络块的概念
    - 块（block）可以描述单个层、由多个层组成的组件或整个模型本身
- 多个层被组合成块，形成更大的模型![[00 Attachments/Pasted image 20240530154942.png|400]]
- 块可以由类表示
    - 它的任何子类都必须定义一个将其输入转换为输出的前向传播函数， 并且必须存储任何必需的参数。
        - 注意，有些块不需要任何参数
    - 最后，为了计算梯度，块必须具有反向传播函数
- 回顾多层感知机的实现

```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

X = torch.rand(2, 20)
print(net(X))
# tensor([[ 0.0819, -0.0157,  0.0066,  0.1114, -0.0334,  0.0681,  0.0723, -0.0512,  
#          -0.0625, -0.0230],  
#         [-0.0059, -0.1143,  0.1253,  0.1412,  0.0894,  0.1311,  0.0627, -0.0428,  
#          -0.2419, -0.0045]], grad_fn=<AddmmBackward0>)
```

- 在这个例子中，我们通过实例化`nn.Sequential`来构建我们的模型， 层的执行顺序是作为参数传递的。 简而言之，`nn.Sequential`
  定义了一种特殊的`Module`， 即在PyTorch中表示一个块的类， 它维护了一个由`Module`组成的有序列表。 注意，两个全连接层都是
  `Linear`类的实例，`Linear`类本身就是`Module`的子类。 另外，到目前为止，我们一直在通过`net(X)`调用我们的模型来获得模型的输出。
  这实际上是`net.__call__(X)`的简写。 这个前向传播函数非常简单： 它将列表中的每个块连接在一起，将每个块的输出作为下一个块的输入

### 自定义块

- 每个块必须提供的基本功能：
    1. 将输入数据作为其前向传播函数的参数。
    2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
    3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
    4. 存储和访问前向传播计算所需的参数。
    5. 根据需要初始化模型参数。
- 从零开始编写一个块。 它包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层。
    - 注意，下面的`MLP`类继承了表示块的类。 我们的实现只需要提供我们自己的构造函数（Python中的`__init__`函数）和前向传播函数

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层  
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。  
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）  
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层  
        self.out = nn.Linear(256, 10)  # 输出层  

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出  
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。  
        return self.out(F.relu(self.hidden(X)))


net = MLP()
print(net(X))
# tensor([[ 0.2562,  0.0372,  0.1668,  0.0516, -0.0890, -0.0652, -0.0237,  0.3862,  
#           0.2786,  0.2802],  
#         [ 0.2898,  0.1183,  0.1148,  0.0293,  0.1075, -0.0088, -0.1856,  0.3277,  
#           0.1517,  0.2112]], grad_fn=<AddmmBackward0>)
```

- 定制的`__init__`函数通过`super().__init__()`调用父类的`__init__`函数， 省去了重复编写模版代码的痛苦
- 实例化两个全连接层， 分别为`self.hidden`和`self.out`
- 注意，除非我们实现一个新的运算符， 否则我们不必担心反向传播函数或参数初始化， 系统将自动生成这些
- 块的==优点==：可以子类化块以创建层（如全连接层的类）、 整个模型（如上面的`MLP`类）或具有中等复杂度的各种组件

### 顺序块

- `Sequential` 类的工作方式：`Sequential`的设计是为了把其他模块串起来。 为了构建我们自己的简化的`MySequential`，
  我们只需要定义两个关键函数
    1. 一种将块逐个追加到列表中的函数
    2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“列表”

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员  
            # 变量_modules中。_module的类型是OrderedDict  
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们  
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256),
                   nn.ReLU(),
                   nn.Linear(256, 10))
print(net(X))
# tensor([[-0.1429, -0.2546, -0.0147,  0.0478, -0.0973, -0.0653,  0.0362,  0.1126,  
#           0.2140,  0.0528],  
#         [-0.1651, -0.4223,  0.0700,  0.0756,  0.1032, -0.1010,  0.2794,  0.0988,  
#           0.0485,  0.0246]], grad_fn=<AddmmBackward0>)
```

### 在前向传播函数中执行代码

- 前向传播函数中执行Python的控制流以及数学运算
- 在块中添加常数参数，不参与更新

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变  
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数  
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数  
        X = self.linear(X)
        # 控制流  
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
print(net(X))
# tensor(0.1110, grad_fn=<SumBackward0>)
```

- 混合搭配各种组合块

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))
# tensor(-0.3325, grad_fn=<SumBackward0>)
```

## 参数管理

- 训练后希望获取最终的参数
- 本节中
    - 访问参数，用于调试、诊断和可视化；
    - 参数初始化；
    - 在不同模型组件间共享参数
- 以单隐藏层的多层感知机为例

```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))
```

### 访问参数

- 当通过`Sequential`类定义模型时， 我们可以通过索引来访问模型的任意层。 这就像模型是一个列表一样，每层的参数都在其属性中

```python
# 参数访问  

print(net[2].state_dict())  # # 访问参数，net[2]就是最后一个输出层  
# OrderedDict([('weight', tensor([[ 0.1890,  0.2337,  0.1736,  0.2461, -0.2516,  0.0174,  0.2657, -0.2497]])), ('bias', tensor([0.2529]))])  
# 访问目标参数  
print(type(net[2].bias))  # 访问bias参数类型  
# <class 'torch.nn.parameter.Parameter'>  
print(net[2].bias)  # 访问bias参数值  
# Parameter containing:  
# tensor([0.2826], requires_grad=True)  
print(net[2].bias.data)  # 访问bias参数值，除去其他信息  
# tensor([0.2826])  
# 访问梯度  
print(net[2].weight.grad == None)  # 由于还未进行反向传播，grad为None  
# True  


#  一次性访问所有参数  

print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # *表示解包  
# ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))  
print(*[(name, param.shape) for name, param in net.named_parameters()])  # 0是第一层名字，1是ReLU，它没有参数  
# ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))  


print(net.state_dict()['2.bias'].data)  # 通过名字获取参数  
# tensor([-0.3512])
```

### 从嵌套块中获取参数

- 自定义块

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套  
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
# tensor([[0.0103],  
#         [0.0104]], grad_fn=<AddmmBackward0>)
```

- 打印网络查看结构

```python
print(rgnet)
# Sequential(  
#   (0): Sequential(  
#     (block 0): Sequential(  
#       (0): Linear(in_features=4, out_features=8, bias=True)  
#       (1): ReLU()  
#       (2): Linear(in_features=8, out_features=4, bias=True)  
#       (3): ReLU()  
#     )  
#     (block 1): Sequential(  
#       (0): Linear(in_features=4, out_features=8, bias=True)  
#       (1): ReLU()  
#       (2): Linear(in_features=8, out_features=4, bias=True)  
#       (3): ReLU()  
#     )  
#     (block 2): Sequential(  
#       (0): Linear(in_features=4, out_features=8, bias=True)  
#       (1): ReLU()  
#       (2): Linear(in_features=8, out_features=4, bias=True)  
#       (3): ReLU()  
#     )  
#     (block 3): Sequential(  
#       (0): Linear(in_features=4, out_features=8, bias=True)  
#       (1): ReLU()  
#       (2): Linear(in_features=8, out_features=4, bias=True)  
#       (3): ReLU()  
#     )  
#   )  
#   (1): Linear(in_features=4, out_features=1, bias=True)  
# )
```

- 可以看出是分层嵌套的
- 访问某一层的参数

```python
print(rgnet[0][1][0].bias.data)
# tensor([0.2691, 0.0448, 0.3411, 0.2580, 0.2636, 0.2124, 0.2467, 0.1614])
```

### 参数初始化

#### 内置函数初始化

- 将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 下划线表示把m.weight的值替换掉  
        nn.init.zeros_(m.bias)


net.apply(init_normal)  # 会递归调用 直到所有层都初始化  
print(net[0].weight.data[0], net[0].bias.data[0])
# tensor([-0.0017, -0.0012,  0.0155,  0.0155]) tensor(0.)
```

- 初始化为给定常数

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print(net[0].weight.data[0])
print(net[0].bias.data[0])
# tensor([1., 1., 1., 1.])  
# tensor(0.)
```

- 对某些块应用不同的初始化方法
    - 例如，下面我们使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
# tensor([-0.0078,  0.1388, -0.4263,  0.5073])  
# tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
```

#### 自定义初始化

- 深度学习框架没有提供我们需要的初始化方法，需要自己定义
- 如$$\begin{split}\begin{aligned}
  w \sim \begin{cases}
  U(5, 10) & \text{ 可能性 } \frac{1}{4} \\
  0 & \text{ 可能性 } \frac{1}{2} \\
  U(-10, -5) & \text{ 可能性 } \frac{1}{4}
  \end{cases}
  \end{aligned}\end{split}$$

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])  # 打印名字是啥，形状是啥  
        nn.init.uniform_(m.weight, -10, 10)  # 均匀分布初始化  
        m.weight.data *= m.weight.data.abs() >= 5  # 大于5的权重乘以5 先判断是否大于等于5得到布尔矩阵  


net.apply(my_init)
# Init weight torch.Size([8, 4])  
# Init weight torch.Size([1, 8])  
print(net[0].weight[:2])
# tensor([[-0.0000, 0.0000, 0.0000, 8.4138],  
#         [-0.0000, 0.0000, 0.0000, 9.4350]], grad_fn=<SliceBackward0>)
```

- 直接设置参数

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])
# tensor([42.0000,  1.0000,  1.0000,  9.4138])
```

### 参数绑定

- 在多个层间共享参数
    - 可以定义一个稠密层，然后使用它的参数来设置另一个层的参数

```python
# 我们需要给共享层一个名称，以便可以引用它的参数  
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同  
print(net[2].weight.data[0] == net[4].weight.data[0])
# tensor([True, True, True, True, True, True, True, True])  
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值  
print(net[2].weight.data[0] == net[4].weight.data[0])
# tensor([True, True, True, True, True, True, True, True])
```

- 这个例子表明第三个和第五个神经网络层的参数是绑定的。
    - 它们不仅值相等，而且由相同的张量表示。 因此，如果我们改变其中一个参数，另一个参数也会改变。
- 这里有一个问题：当参数绑定时，梯度会发生什么情况？
    - 答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层 （即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起（第三个反向后，梯度未清零）

## 自定义层

- 可以用创造性的方式组合不同的层，从而设计出适用于各种任务的架构
    - 时我们会遇到或要自己发明一个现在在深度学习框架中还不存在的层。 在这些情况下，必须构建自定义层

```python
import torch
import torch.nn.functional as F
from torch import nn
```

### 不带参数的层

- 只需继承基础层类并实现前向传播功能

```python
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
# tensor([-2., -1.,  0.,  1.,  2.])
```

- 将层作为组件合并到更复杂的模型中

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())
# tensor(1.8626e-09, grad_fn=<MeanBackward0>)
```

### 带参数的层

- 要使参数可以通过训练进行调整

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))  # nn.Parameter使得这些参数加上了梯度  
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


dense = MyLinear(5, 3)
print(dense.weight)
# Parameter containing:  
# tensor([[-1.2242,  0.7284,  0.7146],  
#         [ 1.0319,  0.7506,  1.4014],  
#         [ 0.8037,  1.3523,  0.2606],  
#         [ 0.3087, -0.8171, -0.4299],  
#         [-0.2361, -0.4604, -0.0679]], requires_grad=True)  

# 使用自定义层直接执行正向传播计算  
print(dense(torch.rand(2, 5)))
# tensor([[0.8832, 0.0000, 0.0000],  
#         [1.0869, 0.0000, 0.0000]])  
# 使用自定义层构建模型  
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
# tensor([[1.7961],  
#         [2.0545]])
```

## 读写文件

- 有时我们希望保存训练的模型， 以备将来在各种环境中使用（比如在部署中进行预测）
- 此外，当运行一个耗时较长的训练过程时， 最佳的做法是定期保存中间结果， 以确保在服务器电源被不小心断掉时，我们不会损失几天的计算结果

### 加载和保存张量

- 对于单个张量，我们可以直接调用`load`和`save`函数分别读写它们。 这两个函数都要求我们提供一个名称，`save`要求将要保存的变量作为输入

```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

- 将存储在文件中的数据读回内存

```python
x2 = torch.load('x-file')
# tensor([0, 1, 2, 3])
```

- 存储张量列表并读取回内存

```python
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print(x2, y2)
# tensor([0, 1, 2, 3]) tensor([0., 0., 0., 0.])
```

- 写入或读取从字符串映射到张量的字典。 当我们要读取或写入模型中的所有权重时，这很方便

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)
# {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}
```

### 加载和保存模型参数

- 保存单个权重向量（或其他张量）确实有用， 但是如果我们想保存整个模型，并在以后加载它们， 单独保存每个向量则会变得很麻烦
- 度学习框架提供了内置函数来保存和加载整个网络
    - 需要注意的一个重要细节是，这将保存模型的参数而不是保存整个模型
    - 因为模型本身可以包含任意代码，所以模型本身难以序列化。 因此，为了恢复模型，我们需要用代码生成架构， 然后从磁盘加载参数

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')
```

- 为了恢复模型，实例化了原始多层感知机模型的一个备份
    - 这里不需要随机初始化模型参数，而是直接读取文件中存储的参数

```python
# 实例化了原始多层感知机模型的一个备份。直接读取文件中存储的参数  
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())  # eval模式下，模型的dropout(丢弃法)层不会随机失活 即进入测试模式  
# MLP(  
#   (hidden): Linear(in_features=20, out_features=256, bias=True)  
#   (output): Linear(in_features=256, out_features=10, bias=True)  
# )  

Y_clone = clone(X)
print(Y_clone == Y)
# tensor([[True, True, True, True, True, True, True, True, True, True],  
#         [True, True, True, True, True, True, True, True, True, True]])
```

- 两个实例具有相同的模型参数，在输入相同的`X`时， 两个实例的计算结果应该相同
