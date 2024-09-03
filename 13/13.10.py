import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 内容和风格图像
d2l.set_figsize()
content_img = d2l.Image.open('er.jpg')
d2l.plt.imshow(content_img)

# 新建画布
d2l.plt.figure()
style_img = d2l.Image.open('autumn-oak.jpg')
d2l.plt.imshow(style_img)

# 预处理和后处理
# 预设的RGB平均值和标准差，用于图像的标准化
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])


def preprocess(img, image_shape):
    """将图片转化为适合模型训练的tensor"""
    # 定义图像预处理流程：调整大小、转换为tensor、标准化
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),  # 调整图像大小到指定的image_shape
        torchvision.transforms.ToTensor(),  # 将图像转换为tensor
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])  # 使用预设的均值和标准差对图像进行标准化
    # 对输入的img应用上述的预处理流程，并在最前面添加一个新维度（用于batch size），然后返回结果
    return transforms(img).unsqueeze(0)


def postprocess(img):
    """将模型输出的tensor转换为图片"""
    img = img[0].to(rgb_std.device)
    # 反标准化处理：对img的每个像素乘以标准差并加上均值，并确保结果在[0,1]范围内
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    # 将处理后的tensor转换为PIL图像，并返回
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


# 抽取图像特征
# 加载预训练的VGG19模型
pretrained_net = torchvision.models.vgg19(pretrained=True)

# 定义要从VGG19模型中提取特征的层的索引
# style_layers用于提取样式特征，content_layers用于提取内容特征
# 这里选择的层数代表了不同级别的特征，越小的层数越接近输入，提取的特征越接近图像的局部信息
style_layers, content_layers = [3, 8, 15, 22], [15]

# 根据给定的样式和内容层索引，从预训练的VGG19模型中抽取所需的层，并创建一个新的神经网络
net = nn.Sequential(*[pretrained_net.features[i]
                      for i in range(max(content_layers + style_layers) + 1)])


def extract_features(X, content_layers, style_layers):
    """从指定层提取特征"""
    contents = []
    styles = []
    # 对于网络中的每一层
    for i in range(len(net)):
        X = net[i](X)  # 在该层上运行输入X以提取特征
        if i in style_layers:  # 如果这是一个样式层，将提取的特征添加到样式列表中
            styles.append(X)
        if i in content_layers:  # 如果这是一个内容层，将提取的特征添加到内容列表中
            contents.append(X)
    return contents, styles


def get_contents(image_shape, device):
    """处理内容图像并提取内容特征"""
    # 预处理内容图像并移动到指定设备上
    content_X = preprocess(content_img, image_shape).to(device)
    # 从内容图像中提取内容特征
    content_Y, _ = extract_features(content_X, content_layers, style_layers)
    # 返回预处理后的内容图像（用于后续的图像合成）和从内容图像中提取的内容特征
    return content_X, content_Y


def get_styles(image_shape, device):
    """处理样式图像并提取样式特征"""
    # 预处理样式图像并移动到指定设备上
    style_X = preprocess(style_img, image_shape).to(device)
    # 从样式图像中提取样式特征
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    # 返回预处理后的样式图像（用于后续的图像合成）和从样式图像中提取的样式特征
    return style_X, styles_Y


def content_loss(Y_hat, Y):
    """内容损失函数"""
    # 计算预测的内容特征（Y_hat）与实际的内容特征（Y）之间的均方误差
    return torch.square(Y_hat - Y.detach()).mean()


def gram(X):
    """Gram矩阵函数，计算输入矩阵（X）的Gram矩阵，用于表示样式特征"""
    # 计算通道数和特征数
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    # 将输入矩阵reshape为(通道数, 特征数)的格式
    X = X.reshape((num_channels, n))
    # 计算Gram矩阵，并进行规范化处理
    return torch.matmul(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    """样式损失函数"""
    # 计算预测的样式特征（Y_hat）的Gram矩阵与实际的样式特征（gram_Y）的Gram矩阵之间的均方误差
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()


def tv_loss(Y_hat):
    """全变分损失函数，用于提高生成图像的空间连续性，以减少生成图像的高频噪声"""
    # 计算图像中相邻像素之间的差值的绝对值的平均值
    # 分别获取张量在高度方向上错开的两个子张量，然后计算它们的差值
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


# 总的损失函数
# 定义内容损失、样式损失和总变差损失的权重
content_weight, style_weight, tv_weight = 1, 1e3, 10


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    """总的损失函数"""
    # 对于每一层的内容特征，计算预测的内容特征（contents_Y_hat）与实际的内容特征（contents_Y）之间的内容损失，
    contents_l = [
        content_loss(Y_hat, Y) * content_weight
        for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    # 对于每一层的样式特征，计算预测的样式特征（styles_Y_hat）的Gram矩阵与实际的样式特征（styles_Y_gram）的Gram矩阵之间的样式损失，
    styles_l = [
        style_loss(Y_hat, Y) * style_weight
        for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    # 计算总变差损失
    tv_l = tv_loss(X) * tv_weight
    # 计算总损失，它是所有层的内容损失、样式损失和总变差损失的加权和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


# 初始化合成图像
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        """调用父类nn.Module的构造函数"""
        super(SynthesizedImage, self).__init__(**kwargs)  # 调用父类构造函数
        # 初始化图像的参数，图像的形状为img_shape，参数的初始值是随机生成的
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        # 直接返回图像的权重
        return self.weight


def get_inits(X, device, lr, styles_Y):
    """初始化合成图像、样式特征的Gram矩阵和优化器"""
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)  # 直接将内容图像的数据复制到gen_img的权重中
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]  # 计算样式特征的Gram矩阵
    return gen_img(), styles_Y_gram, trainer


# 训练模型
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)  # X 为合成图像
    # 定义学习率调度器，每隔lr_decay_epoch个epoch，学习率乘以0.8
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    # 创建一个动画展示器，用于展示每个epoch的内容损失、样式损失和TV损失
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'], ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()  # 清零优化器的梯度
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()  # 计算梯度
        trainer.step()  # 使用梯度更新模型参数
        scheduler.step()  # 更新学习率
        if (epoch + 1) % 10 == 0:
            # 在animator的第二个子图上显示当前生成的图像
            # postprocess函数将生成的图像从张量转换回PIL图像
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1,
                         [float(sum(contents_l)),
                          float(sum(styles_l)),
                          float(tv_l)])
    return X


device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)  # # 对内容图像进行预处理，并提取其内容特征
_, styles_Y = get_styles(image_shape, device)  # 对样式图像进行预处理，并提取其样式特征
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
