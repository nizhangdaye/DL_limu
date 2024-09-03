import sys
import torch
import torchvision  # pytorch自带的图像处理库
from torch.utils import data  # 数据集加载库
from torchvision import transforms  # 图像预处理库
from d2l import torch as d2l

d2l.use_svg_display()  # 使用svg格式显示绘图

# 读取数据集
trans = transforms.ToTensor()  # 转换为张量
# 训练集 train=True下载训练集，transform=trans得到的是pytorch的tensor格式，download=True默认从网上下载
# 如果本地已经有了就不用下载了
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True)  # 训练集
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True)  # 测试集 用于验证模型的好坏

print(len(mnist_train), len(mnist_test))
# 60000 10000
print(mnist_train.data.shape, mnist_test.data.shape)
# torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
print(mnist_train[0][0].shape, mnist_train[0][1])  # 第一个样本的形状和标签


# torch.Size([1, 28, 28]) 9

def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()  # 显示图像
    return axes


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))  # 取出一组样本
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))  # 绘制图像列表

# 读取小批量数据
batch_size = 256


def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,  # True表示每个epoch打乱数据
                             num_workers=get_dataloader_workers()
                             if sys.platform.startswith('win64') else 0)  # windows下num_workers>0可能会报错

# 查看读取数据所用时间
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
# 4.53 sec

# 整合所有组件
def load_data_fashion_mnist(batch_size, resize=None):  #@save  # resize=None表示不改变图像大小
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]  # 定义图像转换操作
    if resize:
        trans.insert(0, transforms.Resize(resize))  # 如果resize不为None，则在图像转换操作中插入Resize操作
    trans = transforms.Compose(trans)  # 组合图像转换操作
    mnist_train = torchvision.datasets.FashionMNIST(  # 下载Fashion-MNIST训练数据集
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(  # 下载Fashion-MNIST测试数据集
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,  # 返回训练数据集和测试数据集的数据加载器
                            num_workers=get_dataloader_workers()
                            if sys.platform.startswith('win64') else 0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()
                            if sys.platform.startswith('win64') else 0))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
# torch.Size([32, 1, 64, 64]) torch.float32 torch.Size([32]) torch.int64
