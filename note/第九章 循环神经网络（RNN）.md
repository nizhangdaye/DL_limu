```toc
```

## 序列模型

- 为解决含有序列数据的一类问题而建立的数学模型

### 序列数据

- 之前的模型考虑的是空间信息（图片），现在
- ![[00 Attachments/Pasted image 20240716101020.png|400]]
- ![[00 Attachments/Pasted image 20240716101103.png|400]]

### 统计工具

-

在图片处理中，每张图片都可认为是相互独立的，但是在序列模型中，我们假设==样本是不独立的==![[00 Attachments/Pasted image 20240716103850.png|400]]

- 联合概率可展开为条件概率
- 所有的机器学习都是对 $P(X)$ （事件 X 发生的概率）进行建模

- 概率的乘法公式![[00 Attachments/Pasted image 20240716110444.png|400]]
    - $$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}, \ldots, x_1)$$
    - 求 $x_T$ 发生的概率（与之前的条件有关），就需要知道之前所有事件发生的概率
        - 在已知发生事件 $(x_1,...,x_{t-1})$ 后，可由条件概率预测 $x_t$
          发生的概率$$x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$$
            - 之后的内容将==围绕如何有效的估计 $P(x_t \mid x_{t-1}, \ldots, x_1)$ 展开==
    - 也可以反过来，用来预测之前事件发生的概率
        - 从信号与系统的角度分析，如果该系统是非因果的

### 序列模型

- 问题的核心在与 $x_1,...,x_{t-1}$ 事件发生后，$x_t$ 发生的概率![[00 Attachments/Pasted image 20240716152833.png|400]]
    - 对之前 t-1 个事件进行建模，用 $f$ 表示
    - 即基于 t-1 个数据（事件）训练模型，然后用模型预测 $x_t$（用自己之前的值预测之后的值，==自回归==）

#### 方案一：马尔科夫模型

- 如果将之前的所用数据都用于预测，将会导致计算量增加![[00 Attachments/Pasted image 20240716192345.png|400]]
    - 对之前 $τ$ 个数据进行建模（使用 $τ$ 个数据去近似 $t$ 个数据）
        - 给定长度为 $τ$ 的向量（特征），预测一个标量
        - 可以视作一个简单的线性回归问题

#### 方案二：潜变量模型

- 使用潜变量概括历史信息（之前的数据）![[00 Attachments/Pasted image 20240716195520.png|400]]
    - $$h_t = f_1(x_{t},h_{t-1})，x_{t+1} = f_2(x_t,h_{t})$$
    - 这样就用两个模型 $f_1,f_2$ ，每个模型只跟两个变量有关

### 总结

- ![[00 Attachments/Pasted image 20240716195745.png|400]]
    - RNN 即为潜变量模型

### 代码实现

- 使用马尔科夫模型来训练一个 MLP（多层感知机）

#### 训练

- 使用正弦函数加上噪声生成序列数据

```python
import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # 总共产生1000个点  
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

- ![[00 Attachments/Pasted image 20240716213037.png|400]]
- 将这个序列转换为模型的特征－标签（feature-label）对
    - 基于嵌入维度 $𝜏$，我们将数据映射为数据对 $𝑦_𝑡=𝑥_𝑡$ 和 $\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$

```python
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
```

- 模型：两个全连接层的错层感知机、RuLU 激活函数和平方损失

```python
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
```

- 训练模型

```python
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
```

#### 预测

##### 单步预测（one-step-adead perdiction）

- 给定前四个数据，预测下一个数据

```python
# 预测模型  
onestep_preds = net(features)
# 进行数据可视化，将真实数据和一步预测结果绘制在同一个图中进行比较  
d2l.plt.figure()
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time', 'x',
         legend=['data', 'l-step preds'], xlim=[1, 1000], figsize=(6, 3))
```

- ![[00 Attachments/Pasted image 20240716213123.png|400]]

##### 多步预测

- 如果数据观察序列的时间步只到 604，
  我们需要一步一步地向前迈进：==在预测的基础上进行预测==$$\begin{split}\hat{x}_{605} = f(x_{601}, x_{602}, x_{603}, x_{604}), \\
  \hat{x}_{606} = f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
  \hat{x}_{607} = f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
  \hat{x}_{608} = f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
  \hat{x}_{609} = f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
  \ldots\end{split}$$
    - 通常，==对于直到 $𝑥_𝑡$ 的观测序列，其在时间步 $𝑡+𝑘$ 处的预测输出 $\hat{x}_{t+k}$称为 $𝑘$
      步预测==（𝑘-step-ahead-prediction）
    - 由于我们的观察已经到了 $𝑥_{604}$，它的 $𝑘$ 步预测是 $\hat{x}_{604+k}$
        - 换句话说，我们必须使用我们自己的预测（而不是原始数据）来进行多步预测

```python
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
```

- 可以发现预测发生较大的偏移（前几步预测可以，但是之后的 k
  步预测不行）：![[00 Attachments/Pasted image 20240716213207.png|400]]
    - 紫色部分是用实际的数据进行预测
    - 绿色部分使用预测的数据进行预测
        - 经过几个预测步骤之后，预测的结果很快就会衰减到一个常数
        - 将误差进行了不断的累计
- 基于𝑘=1,4,16,64，通过对整个序列预测的计算， 更仔细地看一下 𝑘 步预测的困难

```python
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
```

- 这张图体现了多步预测的困难![[00 Attachments/Pasted image 20240716213254.png|400]]
    - 即使这是一个很简单的函数，也很难预测很远的未来
- 之后的内容围绕如何预测更远的未来

## 文本预处理

- 相关名词：
    - ==词元==（Token）：词元是文本的基本单元。它可以是单词、标点符号、子单词或字符，具体取决于分词（tokenization）策略
    - ==语料库==（Corpus）：语料库是经过整理和标注的大量文本集合，用于语言研究和NLP模型的训练
        - 例如，常见的语料库包括新闻文章、书籍、社交媒体帖子等。语料库的质量和规模对NLP模型的性能有直接影响
        - 语料库通常需要经过清理、标注等预处理步骤，以提高其有效性
    - ==词汇表==（Vocabulary）：词汇表是从语料库中提取的所有独特词元的集合
        - 它包含了语料库中出现的所有词元及其频率或其他统计信息

- 文本预处理的常见步骤
    1. 将文本作为字符串加载到内存中
    2. 将字符串拆分为词元（如单词和字符）
    3. 构建语料库
    4. 建立一个词表，将拆分的词元映射到数字索引
    5. ==将文本转换为数字索引序列==，方便模型操作

### 读取数据集

- 将数据集读取到由多条文本行组成的列表中，其中每条文本行都是一个字符串
    - 此处忽略标点和大写

```python
import collections
import re
from d2l import torch as d2l

# 读取数据集  
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():
    """将时间机器数据集加载为文本行的列表"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()  # 读取所有行  
        # 把不是大写字母、小写字母的东西，全部变成空格  
        # 去除非字母字符，并转换为小写  
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]  # 正则表达式  


# 读取时间机器数据集，并将结果存储在 'lines' 变量中  
lines = read_time_machine()
print(lines[0])
print(lines[10])
# the time machine by h g wells  
# twinkled and his usually pale face was flushed and animated the
```

### 词元化

- `tokenize`函数将文本行列表（`lines`）作为输入，列表中的每个元素是一个文本序列（如一条文本行）
    - 每个文本序列又被拆分成一个词元列表，==词元==（token）是文本的基本单位
    - 最后，返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）

```python
def tokenize(lines, token='word'):
    """将文本行列表进行分词处理"""
    if token == 'word':
        # 以空格为分隔符将每行字符串拆分为单词列表  
        return [line.split() for line in lines]
    elif token == 'char':
        # 将每行字符串拆分为字符列表（包括空格）
        return [list(line) for line in lines]
    else:
        print('错位：未知令牌类型：' + token)


tokens = tokenize(lines)  # 对 lines 进行分词处理  
for i in range(11):
    # 空列表表示空行  
    print(tokens[i])
# ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']  
# []  
# []  
# []  
# []  
# ['i']  
# []  
# []  
# ['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him']  
# ['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']  
# ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
```

### 词表（vocabulary）

- 词元的类型是字符串，但是模型的输入是数字
- 因此需要构建一个词表（字典），将字符串类型的词元映射到从 0 开始的数字索引中
    - 先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计，得到的统计结果称之为==语料==（corpus）
    - 然后根据每个唯一词元的出现频率，为其分配一个数字索引
        - 很少出现的词元通常被移除，这可以降低复杂性
        - 另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“$<unk>$”
        - 可以选择增加一个列表，用于保存那些被保留的词元
            - 例如：填充词元（“$<pad>$”）； 序列开始词元（“$<bos>$”）； 序列结束词元（“$<eos>$”）

```python
class Vocab:
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """初始化词表对象"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)  # 统计 tokens 中词元的频率  
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 设置未知标记索引为 0，构建包含未知标记和保留特殊标记的列表 uniq_tokens
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]  # 保留词频大于等于 min_freq 的标记  
        # token_to_idx 获取词元到索引的映射，idx_to_token 获取索引到词元的映射(词表)  
        self.idx_to_token, self.token_to_idx = [], dict()
        # 遍历 uniq_tokens 中的每个标记，将其添加到索引到标记的列表中，并将标记和对应索引存储到标记到索引的字典中  
        # 索引值从 0 开始递增，对应于标记在列表中的位置  
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            # 将当前标记 `token` 和其对应的索引值存储到标记到索引的字典 `self.token_to_idx` 中  
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """获取词表的长度"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """根据标记获取其对应的索引或索引列表"""
        # 如果 tokens 不是列表或元组，则返回对应的索引或默认的未知标记索引  
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
            # 对于输入的标记列表 tokens，逐个调用 self.__getitem__() 方法获取每个标记对应的索引值，并返回索引值的列表    
            return [self.__getitem__(token) for token in tokens]


def to_tokens(self, indices):
    """根据索引获取对应的标记或标记列表"""
    # 如果输入的 indices 不是列表或元组类型，则返回对应索引值处的标记  
    if not isinstance(indices, (list, tuple)):
        return self.idx_to_token[indices]
    return [self.idx_to_token[index] for index in indices]


def count_corpus(tokens):
    """统计标记的频率"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 如果 tokens 是一个列表的列表，则将其展平为一维列表  
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)  # 使用 Counter 对象统计词元的频率
```

- 使用时光机器数据集作为语料库来构建词表，然后打印前几个高频词元及其索引

```python
# 构建词汇表  
vocab = Vocab(tokens)  # 创建词表  
print(list(vocab.token_to_idx.items())[:10])
# [('<unk>', 0), ('the', 1), ('i', 2), ('and', 3), ('of', 4), ('a', 5), ('to', 6), ('was', 7), ('in', 8), ('that', 9)]
```

- 将每一条文本行转换成一个数字索引列表

```python
# 将每一行文本转换成一个数字索引列表  
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])
# 文本: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']  
# 索引: [1, 19, 50, 40, 2183, 2184, 400]  
# 文本: ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']  
# 索引: [2186, 3, 25, 1044, 362, 113, 7, 1421, 3, 1045, 1]
```

### 整合所有功能

- `load_corpus_time_machine`函数返回`corpus`（词元索引列表）和`vocab`（时光机器语料库的词表）
    1. 为了简化后面章节中的训练，使用字符（而不是单词）实现文本词元化（为什么使用字符可以简化，使用字符的话，词表不超过 28）
    2. 时光机器数据集中的每个文本行不一定是一个句子或一个段落，还可能是一个单词，因此返回的`corpus`
       仅处理为单个列表，而不是使用多词元列表构成的一个列表（？？？？）

```python
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的标记索引列表和词汇表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')  # 以字符为单位进行分词  
    vocab = Vocab(tokens)  # 创建词表  
    # 将文本转换为标记索引列表  
    corpus = [vocab[token] for line in tokens for token in line]
    # 截断文本长度（若有限制）  
    if max_tokens > 0:
        # 对标记索引列表 corpus 进行截断，只保留前 max_tokens 个标记  
        corpus = corpus[:max_tokens]
    return corpus, vocab


# 载入时光机器数据集的标记索引列表和词汇表  
corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
# 170580 28
print(vocab.token_to_idx.keys())
# dict_keys(['<unk>', ' ', 'e', 't', 'a', 'i', 'n', 'o', 's', 'h', 'r', 'd', 'l', 'm', 'u', 'c', 'f', 'w', 'g', 'y', 'p', 'b', 'v', 'k', 'x', 'z', 'j', 'q'])
```

## 语言模型和数据集

- 用于预测文本序列中下一个词或字符的概率分布的模型![[00 Attachments/Pasted image 20240717163109.png|400]]

### 使用计数建模

- 一个简单的建模方式![[00 Attachments/Pasted image 20240717164757.png|400]]
    - 连续出现的概率

### 马尔可夫模型与 𝑛 元语法

- 但是在计数建模中，如果采用的文本序列过大，很可能导致 n（该连续文本出现的次数）为
  0![[00 Attachments/Pasted image 20240717165406.png|400]]
    - 一元语法：假设每个词是独立的，只依赖自己（即在马尔科夫模型中 τ 为 0）
    - 二元语法：假设每个词只依赖于前一个词（即在马尔科夫模型中 τ 为 1）
        - 计数 $n(x_i, x_{x+1})$ 时只需寻找长度为 2 的子序列
- N 越大，对应的依赖关系越长，精度越高，但是空间复杂度比较大

### 总结

- 语言模型估计文本序列的联合概率
- 使用统计方法时常采用 n 元语法
    - 每次找一个长为 n 的子序列，然后去计数，用于计算概率

### 代码实现

#### 自然语言统计

- 对时光机器构建词表，并打印前 10 个频率最高的词

```python
import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())  # 进行分词处理  
corpus = [token for line in tokens for token in line]  # 将所有单词拼接成一个列表，生成语料库  
vocab = d2l.Vocab(corpus)  # 使用语料库构建词汇表  
print(vocab.token_freqs[:10])
# [('the', 2261),  
#  ('i', 1267),  
#  ('and', 1245),  
#  ('of', 1155),  
#  ('a', 816),  
#  ('to', 695),  
#  ('was', 552),  
#  ('in', 541),  
#  ('that', 443),  
#  ('my', 440)]
```

- 一些常见的流行词通常被称为==停用词==（stop words），因此可以被过滤，但是依然会在模型中使用（多个单词组合）
- 可以看到单词的使用频率下降的很快，画出词频图

```python
# 从词汇表的token_freqs中提取频率信息，存储在列表freqs中  
freqs = [freq for token, freq in vocab.token_freqs]
# 使用d2l库中的plot函数绘制词频图  
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')
```

- ![[00 Attachments/Pasted image 20240729194200.png|400]]
- 通过图片可以发现：词频以近似线性的方式迅速衰减
- 接下来查看二元语法、三元语法的情况

```python
# 二元语法  
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]  # 二元语法语料库  
bigram_vocab = d2l.Vocab(bigram_tokens)  # 构建词汇表  
print(bigram_vocab.token_freqs[:10])
# [(('of', 'the'), 309),  
#  (('in', 'the'), 169),  
#  (('i', 'had'), 130),  
#  (('i', 'was'), 112),  
#  (('and', 'the'), 109),  
#  (('the', 'time'), 102),  
#  (('it', 'was'), 99),  
#  (('to', 'the'), 85),  
#  (('as', 'i'), 78),  
#  (('of', 'a'), 73)]  
# 三元语法  
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)  # 构建词汇表  
for i in trigram_vocab.token_freqs[:10]:
    print(i)
# (('the', 'time', 'traveller'), 59)  
# (('the', 'time', 'machine'), 30)  
# (('the', 'medical', 'man'), 24)  
# (('it', 'seemed', 'to'), 16)  
# (('it', 'was', 'a'), 15)  
# (('here', 'and', 'there'), 15)  
# (('seemed', 'to', 'me'), 14)  
# (('i', 'did', 'not'), 14)  
# (('i', 'saw', 'the'), 13)  
# (('i', 'began', 'to'), 13)
```

- 可以看到当使用三元语法的时候高频词更能反应文章的信息（信息量更多）
- 画出这三种词元语法的词频图

```python
# 直观地对比三种模型中的标记频率  
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

- ![[00 Attachments/Pasted image 20240729194255.png|400]]
- 使用 n 元语法，当 n 越大，计算量也越大（指数关系），但是从图中也可以看出，在三元语法中，大多数词元的出现频率非常小，可以进行过滤操作（一刀切）
    - ==长序列存在一个问题：它们很少出现或者从不出现==
    - 所以在实际中，n 取较大值也是可以接受的

#### 读取长序列数据（两种读取方式）

- 序列数据本质上是连续的，因此当序列变得太长而不能被模型一次性全部处理时，就希望对序列进行拆分以方便模型的读取
- 策略
    - 假设使用神经网络来训练语言模型，模型中的网络一次处理具有预定义长度（n 个时间步）的一个小批量序列
- 那么该如何随机生成一个小批量数据的特征和标签一共读取？（类似滑动窗口）
    - 首先，文本序列是任意长的，因此任意长的序列可以被划分为具有==相同（n）时间步长的子序列==
    - 假设神经网络一次处理具有 n 个时间步的子序列
    - 从原始文本中获取子序列，起始位置不同![[00 Attachments/Pasted image 20240729200812.png|400]]
    - 使用随机偏移量来指定子序列的起始位置
        - 覆盖性（coverage）：取得的子序列尽可能覆盖原始文本
        - 随机性（randomness）：子序列的起始位置是随机的

##### 随机采样（random sampling）

- 在随机采样中，每个样本都是在原始的长序列上任意捕获的子序列
- 在迭代过程中，来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
- 对于语言建模，目标是基于到目前为止我们看到的词元来预测下一个词元，因此标签是移位了一个词元的原始序列

```python
# 随即生成一个小批量数据的特征和标签以供读取  
# 在随即采样中，每个样本都是在原始的长序列上任意捕获的子序列  

# 给一段很长的序列，连续切成很多段长为T的子序列  
# 一开始加了一点随机，使得每次切的起始位置不同  
def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随即抽样生成一个小批量子序列  
    :param corpus: 词汇表  
    :param batch_size: 批量大小  
    :param num_steps: 子序列长度  
    :return: 一个生成器，每个元素是一批子序列的特征和标签  
    """  # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1 确定起始位置  
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 计算能够生成的子序列数量  
    num_subseqs = (len(corpus) - 1) // num_steps
    # 创建初始索引列表  
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 进行随机打乱  
    random.shuffle(initial_indices)

    # 返回从指定位置开始的长度为num_steps的子序列  
    def data(pos):
        return corpus[pos:pos + num_steps]

        # 计算批次的数量  

    num_batches = num_subseqs // batch_size
    # 对每个批次进行迭代  
    for i in range(0, batch_size * num_batches, batch_size):
        # 获取当前批次的初始索引列表  
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        # 根据初始索引列表获取对应的特征序列X  
        X = [data(j) for j in initial_indices_per_batch]
        # 根据初始索引列表获取对应的标签序列Y  
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # 使用torch.tensor将X和Y转换为张量，并通过yield语句返回  
        yield torch.tensor(X), torch.tensor(Y)
```

- 生成一个从 0 到 34 的序列
    - 假设批量大小为2，时间步数为5，这意味着可以生成$\lfloor (35 - 1) / 5 \rfloor= 6$ 个“特征－标签”子序列对
    - 如果设置小批量大小为2，我们只能得到3个小批量

```python
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
# X:  tensor([[ 5,  6,  7,  8,  9],  
#         [15, 16, 17, 18, 19]])
# Y: tensor([[ 6,  7,  8,  9, 10],  
#         [16, 17, 18, 19, 20]])  
# X:  tensor([[25, 26, 27, 28, 29],  
#         [ 0,  1,  2,  3,  4]])
# Y: tensor([[26, 27, 28, 29, 30],  
#         [ 1,  2,  3,  4,  5]])  
# X:  tensor([[10, 11, 12, 13, 14],  
#         [20, 21, 22, 23, 24]])
# Y: tensor([[11, 12, 13, 14, 15],  
#         [21, 22, 23, 24, 25]])
```

##### 顺序分区（sequential partitioning）

- 在迭代过程中，除了对原始序列可以随机抽样外，我们还可以保证==两个相邻的小批量中的子序列在原始序列上也是相邻的==（不是小批量中的子序列相邻）
- 这种策略在基于小批量的迭代过程中==保留了拆分的子序列的顺序==，因此称为顺序分区

```python
# 保证两个相邻的小批量中的子序列在原始序列上也是相邻的  
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """  
    使用顺序分区生成一个小批量子序列  
    :param corpus: 词汇表  
    :param batch_size: 批量大小  
    :param num_steps: 子序列长度  
    :return: 一个生成器，每个元素是一批子序列的特征和标签  
    """  # 随机选择一个偏移量作为起始位置  
    offset = random.randint(0, num_steps)
    # 计算可以生成的子序列的总长度  
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    # 创建特征序列X的张量  
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    # 创建标签序列Y的张量  
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    # 重新调整Xs和Ys的形状，使其成为(batch_size, -1)的二维张量  
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    # 计算可以生成的批次数量  
    num_batches = Xs.shape[1] // num_steps
    # 对每个批次进行迭代  
    for i in range(0, num_steps * num_batches, num_steps):
        # 获取当前批次的特征序列X  
        X = Xs[:, i:i + num_steps]
        # 获取当前批次的标签序列Y  
        Y = Ys[:, i:i + num_steps]
        # 使用yield语句返回X和Y作为生成器的输出  
        yield X, Y


for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
# X:  tensor([[ 5,  6,  7,  8,  9],  
#         [19, 20, 21, 22, 23]]) 
# Y: tensor([[ 6,  7,  8,  9, 10],  
#         [20, 21, 22, 23, 24]])  
# X:  tensor([[10, 11, 12, 13, 14],  
#         [24, 25, 26, 27, 28]])
# Y: tensor([[11, 12, 13, 14, 15],  
#         [25, 26, 27, 28, 29]])
```

- 迭代期间来自两个相邻的小批量中的子序列在原始序列中确实是相邻的

##### 打包

```python
class SeDataLoader:
    """加载序列数据的迭代器"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        # 根据use_random_iter选择数据迭代函数  
        if use_random_iter:
            # 使用随机分区迭代器  
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            # 使用顺序分区迭代器  
            self.data_iter_fn = d2l.seq_data_iter_sequential
            # 加载数据集和词汇表  
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        # 设置批量大小和步长  
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        # 返回数据迭代器  
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词汇表"""
    # 这个对象将作为数据的迭代器，用于产生小批量的样本和标签。  
    data_iter = SeDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    # 返回数据迭代器和对应的词汇表  
    return data_iter, data_iter.vocab
```

## 循环神经网络 （recurrent neural networks）

### 循环神经网络

- 标准神经网络中的所有输入和输出都是相互独立的，但是在某些情况下，例如在预测短语的下一个单词时，前面的单词是必要的，因此必须记住前面的单词
    - 结果，RNN 应运而生，它使用隐藏层来克服这个问题。 RNN 最重要的组成部分是隐藏状态，它记住有关序列的特定信息

#### 带隐状态的循环神经网络

- 回顾潜变量模型![[00 Attachments/Pasted image 20240726004026.png|400]]
- 在多层感知机中加入潜变量，==捕获并保留了序列直到其当前时间步的历史信息==，
  就如当前时间步下神经网络的状态或记忆![[00 Attachments/Pasted image 20240726010827.png|400]]
    - $h_t$ 由之前的 x 确定，用于预测当前的 $x_t$（即 $o_t$）
    - 实际的效果（打字预测）![[00 Attachments/Pasted image 20240726010227.png|400]]
    - 损失是通过 $o_t$ 和 $x_t$ 的关系来计算 $loss = f(x_t, o_t)$
    - ==保存了前一个时间步的隐藏变量 $\mathbf{H}_{t-1}$，
      并引入了一个新的权重参数 $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$， 来描述如何在当前时间步中使用前一个时间步的隐藏变量==
- 隐状态使用的定义与前一个时间步中使用的定义相同， 因此潜变量的计算是循环的（recurrent）
    - 于是基于循环计算的隐状态神经网络被命名为循环神经网络（recurrent neural network）
- 三个相邻时间步的计算逻辑![[00 Attachments/Pasted image 20240726013627.png|400]]
    1. 拼接当前时间步 𝑡 的输入 𝑋𝑡 和前一时间步 𝑡−1 的隐状态 𝐻𝑡−1
    2. 将拼接的结果送入带有激活函数 𝜙 的全连接层。 全连接层的输出是当前时间步 𝑡 的隐状态 𝐻𝑡

#### 困惑度（Perplexity）

- 衡量模型的好坏![[00 Attachments/Pasted image 20240726013828.png|400]]
    - 输出即为判断下一个词，假设词表中有 m 个词，==一个词就可以视为一个类别，即类似于做一个分类问题==，判断各个词出现的概率，那么就可以使用交叉熵损失
        - 通过一个序列中所有的 𝑛 个词元的交叉熵损失的平均值来衡量
        - 𝑃 由语言模型给出， 𝑥𝑡 是在时间步 𝑡 从该序列中观察到的实际词元
- 困惑度的最好的理解是“下一个词元的实际选择数的调和平均数”
    - 在最好的情况下，模型总是完美地估计标签词元的概率为 1
        - 在这种情况下，模型的困惑度为 1
    - 在最坏的情况下，模型总是预测标签词元的概率为 0
        - 在这种情况下，困惑度是正无穷大
    - 在基线上，该模型的预测是词表的所有可用词元上的均匀分布
        - 在这种情况下，困惑度等于词表中唯一词元的数量
        - 事实上，如果我们在没有任何压缩的情况下存储序列， 这将是我们能做的最好的编码方式
        - 因此，这种方式提供了一个重要的上限， 而任何实际模型都必须超越这个上限

#### 梯度裁剪

- 对于长度为 T 的序列，在迭代中计算这T个时间步上的梯度， 将会在反向传播过程中产生长度为O(T)的矩阵乘法链
    - 当 T 较大时，它可能导致数值不稳定， 例如可能导致梯度爆炸或梯度消失
- 控制梯度的数值![[00 Attachments/Pasted image 20240726020146.png|400]]
    - 梯度链：通过平均交叉熵损失求梯度，沿着步长反向传播：$t——>t-1——>t-2——>...$

#### RNN 的应用

- ![[00 Attachments/Pasted image 20240726020645.png|400]]
    - 所谓的一对一就是 MLP，给一个样本，输出一个预测类别
    - 文本生成：给定开始的词（图片或者音乐），预测下一个词，将预测的词作为输入，用于继续预测（看图说话）
    - 文本分类：给定句子序列，随着序列的输入更新隐变量层，最后将分本分类（情感分类）
    - Tag 生成：词性预测

#### 总结

- 循环神经网络的输出取决于当下输入和前一时间的隐变量
    - 隐变量用于==存储历史时刻的信息==
- 应用到语言模型中时，循环神经网络根据当前词预测下一次时刻词
- 通常使用困惑度来衡量语言模型的好坏
    - 每一步预测的平均
- ![[00 Attachments/Pasted image 20240726081005.png|500]]

### RNN 从零开始实现

- 读取数据集

```python
%matplotlib
inline
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 定义批量大小和时间步数
batch_size, num_steps = 32, 35
# 加载时间机器数据并创建词汇表
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

#### 独热编码

- ==通过独热编码将每个词元（已经完成到数字的映射）表示为更具表现力的特征向量==
    - 之后会使用嵌入层来表示词元之间的关系，意思相近的词元通过嵌入层后的空间位置会更加相近
- 举例

```python
# 打印词汇表的大小  
print(len(vocab))
# 使用独热编码将 [0, 2] 表示的物体下标转换为独热向量，其中0表示第一个元素，2表示第3个元素  
F.one_hot(torch.tensor([0, 2]), len(vocab))
# 28  
# tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
#          0, 0, 0, 0],  
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
#          0, 0, 0, 0]])
```

- 采样的小批量数据的形状是 （批量大小、时间步数），为二维张量
    - one_hot 将这个批量数据转化为三维张量（批量大小、时间步数、词汇标大小）
    - 能够更方便地通过最外层的维度， 一步一步地更新小批量数据的隐状态

```python
# 小批量形状是(批量大小，时间步数)  
X = torch.arange(10).reshape((2, 5))  # 对X的转置进行独热编码，其中28表示编码长度，返回独热编码后的形状  
F.one_hot(X.T, 28).shape
# torch.Size([5, 2, 28])
```

- 这里的转置是为了将时间步为行，批量为列，方便按时间步进行索引（各个批量是有序的）

#### 初始化模型参数

- 隐藏单元数`num_hiddens`是一个可调的超参数。 当训练语言模型时，输入和输出来自相同的词表。 因此，它们具有相同的维度，即词表的大小

```python
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
```

#### 循环神经网络模型

- 在定义神经网络模型之前，首先需要一个 `init_rnn_state` 初始化隐状态

```python
def init_rnn_state(batch_size, num_hiddens, device):
    """初始化为零的隐藏状态"""
    return (torch.zeros((batch_size, num_hiddens), device=device),)
```

- 下面的`rnn`函数定义了如何在一个时间步内计算隐状态和输出
    - 循环神经网络模型通过`inputs`最外层的维度（时间步）实现循环， 以便逐时间步更新小批量数据的隐状态`H`
- ==类似于前向传播==，但又有些不同
    - `input`不再是（批量，大小），而是（时间步数，批量，大小）
    - 加入了隐状态作为循环的实现

```python
def rnn(inputs, state, params):
    """  
    定义循环神经网络模型，包括了所有时间步的计算，类似于前向传播  
    :param inputs: 输入序列形状为 (时间步数，批量大小, 词汇表大小)  
    :param state: 隐藏状态，形状为 (批量大小, 隐藏层大小)  
    :param params: 模型参数，包含权重矩阵和偏置向量  
    :return: 返回输出序列和新的隐藏状态，形状均为 (时间步数 * 批量大小, 输出大小)  
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
```

- 创建一个类来包装这些函数，并存储从零开始实现的循环神经网络模型的参数

```python
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
```

- 举例，查看输出

```python
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())  # 输入输出都为（批量大小，隐藏层大小）  
Y, new_state = net(X.to(d2l.try_gpu()), state)  # 前向传播
Y.shape, type(new_state), len(new_state), new_state[0].shape
# (torch.Size([10, 28]), tuple, 1, torch.Size([2, 512]))
```

- 可以看到输出形状是（时间步数×批量大小，词汇表大小），而隐状态形状保持不变，即（批量大小，隐藏单元数）

#### 预测

- 定义预测函数来生成`prefix`之后的新字符
    - 在循环遍历`prefix`中的开始字符时，不断地将隐状态传递到下一个时间步，但是不生成任何输出
    - 这被称为预热（warm-up）期，因为在此期间模型会自我更新（例如，更新隐状态），但不会进行预测
    - 预热期结束后，隐状态的值通常比刚开始的初始值更适合预测，从而预测字符并输出它们

```python
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
```

- 尝试预测（还没有训练）

```python
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
# 'time traveller lkxdcccccc'
```

#### 梯度裁剪

- $$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}$$

```python
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
```

#### 训练

- 在训练模型之前，需要定义一个函数在一个迭代周期内训练模型。它与之前的方式有三个不同之处
    1. 序列数据的不同采样方法（随机采样和顺序分区）将导致隐状态初始化的差异
    2. 在更新模型参数之前裁剪梯度
        - 这样的操作的目的是，即使训练过程中某个点上发生了梯度爆炸，也能保证模型不会发散
    3. 采用困惑度来评价模型
        - 这样的度量确保了不同长度的序列具有可比性
- 两种不同分区的处理方式
    - 当使用顺序分区时，==只在每个迭代周期的开始位置初始化隐状态==
        - 由于下一个小批量数据中的第 i 个子序列样本与当前第 i
          个子序列样本相邻（时序上连续），因此==当前小批量数据最后一个样本的隐状态，将用于初始化下一个小批量数据第一个样本的隐状态==（？？？？？）
          ```python
          # X:  tensor([[ 5,  6,  7,  8,  9],  
          #         [19, 20, 21, 22, 23]]) 
          # Y: tensor([[ 6,  7,  8,  9, 10],  
          #         [20, 21, 22, 23, 24]])  
          # X:  tensor([[10, 11, 12, 13, 14],  
          #         [24, 25, 26, 27, 28]])
          # Y: tensor([[11, 12, 13, 14, 15],  
          #         [25, 26, 27, 28, 29]])
          ```
            - 这样，存储在隐状态中的序列的历史信息可以在一个迭代周期内流经相邻的子序列
        - 然而，在任何一点隐状态的计算，都依赖于同一迭代周期中前面所有的小批量数据， 这使得梯度计算变得复杂
            - 为了降低计算量，在处理任何一个小批量数据之前， 先分离梯度，使得隐状态的梯度计算总是限制在一个小批量数据的时间步内
    - 当使用随机抽样时，因为每个样本都是在一个随机位置抽样的，因此==需要为每个迭代周期重新初始化隐状态==
        - 与[3.6节](https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression-scratch.html#sec-softmax-scratch)中的
          `train_epoch_ch3`函数相同，`updater`是更新模型参数的常用函数。 它既可以是从头开始实现的`d2l.sgd`函数，
          也可以是深度学习框架中内置的优化函数

```python
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
    y = Y.T.reshape(-1)
    X, y = X.to(device), y.to(device)
    y_hat, state = net(X, state)  # 前向传播计算预测值和新的隐藏状态  
    l = loss(y_hat, y.long()).mean()  # 计算损失  
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
```

- 从损失函数可以看出，之所以对输出进行 `torch.cat(outputs, dim=0)`，是因为==虽然这是个语言模型，但它是一个标准的多分类问题==
    - 多分类，对一个批量的输入做预测
    - 语言模型，对一个批量不同时间步的输入做预测
- 训练函数，同时满足从零开始实现或是简洁实现

```python
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
```

- 使用顺序分区进行预测

```python
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())  # 循序分区  
# 困惑度 1.0, 45872.6 标记/秒 cuda:0
# time travelleryou can show black is white by argument said filby  
# travelleryou can show black is white by argument said filby
```

- ![[00 Attachments/Pasted image 20240806221957.png|400]]
    - 可以看到这里的困惑都为 1，说明已经完美的将文本进行了记忆
        - 因为这就是一本书，迭代 500 个周期，那么很容易将这本书给背下来了
    - 观察预测的结果，发现从单词、词组、句子的角度来看，是没有问题的，但是到句子的组合，可以看到关联性不大
        - 这也是语言模型最常见的问题，==禁不起细看==
        - 原文：
          ```
          'What reason?' said the Time Traveller.  
            
          'You can show black is white by argument,' said Filby, 'but you will  
          never convince me.'
          ```
- 使用随机抽样分区进行预测

```python
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), use_random_iter=True)  # 随机抽样分区  
# 困惑度 1.5, 45742.9 标记/秒 cuda:0
# time travellerit s against reason said the medical man there are  
# travellerit s against reason said the medical man there are
```

- ![[00 Attachments/Pasted image 20240806223057.png|400]]
    - 可以看到这里的困惑度没有上一个好，这可能是因为这里引入了随机性，训练起来就比较难，没有很好得将整本书记下来
    - 原文
      ```
      'It's against reason,' said Filby.  
        
      'What reason?' said the Time Traveller.
      ```
- 总的来说，就单个词来说，效果还是不差的

### 简洁实现

#### 定义模型

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_hiddens = 256  # 256 个隐藏单元
rnn_layer = nn.RNN(len(vocab), num_hiddens)  # 隐层，返回下一层的输出和新隐态

state = torch.zeros((1, batch_size, num_hiddens))
print(state.shape)
# torch.Size([1, 32, 256])
```

- 通过一个隐状态和一个输入，就可以用更新后的隐状态计算输出
- rnn_layer 的输出 Y 不涉及输出层的计算
    - 它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入

```python
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)  # Y 时间，批量，输出
# (torch.Size([35, 32, 256]), torch.Size([1, 32, 256]))
```

- 定义一个 RNNModel 类
    - 因为 rnn_layer 只包含隐藏的循环层，所以还需要创建一个单独的输出层

```python
class RNNModel(nn.Module):
    """循环神经网络模型"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的，num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)  # 将每一个词转换成独热编码
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)  # 隐藏层
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        """
        状态初始化
        """

    if not isinstance(self.rnn, nn.LSTM):
        # nn.GRU以张量作为隐状态
        return torch.zeros((self.num_directions * self.rnn.num_layers,
                            batch_size, self.num_hiddens),
                           device=device)
    else:
        # nn.LSTM以元组作为隐状态
        return (torch.zeros((self.num_directions * self.rnn.num_layers,
                             batch_size, self.num_hiddens),
                            device=device),
                torch.zeros((self.num_directions * self.rnn.num_layers,
                             batch_size, self.num_hiddens),
                            device=device))
```

#### 预测与训练

- 训练之前的预测

```python
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
# 'time travellerbbabbkabyg'
```

- 可以看到由于没有训练，预测的效果很不好

```python
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)

```

- 
