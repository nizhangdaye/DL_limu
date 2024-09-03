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


# 构建词汇表
vocab = Vocab(tokens)  # 创建词表
print(list(vocab.token_to_idx.items())[:10])
# [('<unk>', 0), ('the', 1), ('i', 2), ('and', 3), ('of', 4), ('a', 5), ('to', 6), ('was', 7), ('in', 8), ('that', 9)]

# 将每一行文本转换成一个数字索引列表
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])


# 文本: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']


# 索引: [1, 19, 50, 40, 2183, 2184, 400]
# 文本: ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
# 索引: [2186, 3, 25, 1044, 362, 113, 7, 1421, 3, 1045, 1]

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
