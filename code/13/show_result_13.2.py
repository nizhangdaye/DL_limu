import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from d2l import torch as d2l
import os

# 获取数据集
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')  # 下载数据集
data_dir = d2l.download_extract('hotdog')  # 下载解压数据集

# 数据预处理
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet数据集的均值和标准差
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),  # 调整图像大小为256×256
    torchvision.transforms.CenterCrop(224),  # 裁剪图像中央的224×224
    torchvision.transforms.ToTensor(),
    normalize])

test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)  # 读取测试集图片

# 选择一个图像并添加batch维度
X = torch.unsqueeze(test_imgs[5][0], 0)
actual_class = test_imgs[5][1]  # 类别名称


# 展示图片
def show_image(img_tensor, title=None):
    img = img_tensor.squeeze(0)  # 移除batch维度
    img = img.permute(1, 2, 0)  # 从C, H, W转换为H, W, C
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


# 定义ResNet18网络
net = torchvision.models.resnet18()
net.fc = nn.Linear(net.fc.in_features, 2)  # 重新定义输出层 2个类别
net.load_state_dict(torch.load('resnet18_13.2.params'))  # 加载预训练模型参数

# 创建一个字典来存储每个卷积层的输出
outputs = {}


# 定义钩子函数并接受层编号
def hook_fn(module, input, output, layer_idx):
    if isinstance(module, nn.Conv2d):
        outputs[(module, layer_idx)] = output


# 注册钩子函数到每个卷积层并添加层编号
layer_idx = 0
for layer in net.modules():  # modules()函数用于返回网络中的所有模块
    if isinstance(layer, nn.Conv2d):  # isinstance()函数用于判断对象是否是一个已知的类型
        layer_idx += 1
        layer.register_forward_hook(lambda module, input, output, idx=layer_idx: hook_fn(module, input, output, idx))

# 前向传播
net.eval()
with torch.no_grad():
    y_hat = net(X)

# 获取预测值
_, predicted = torch.max(y_hat, 1)

# 打印预测结果和实际类别
print(f'Predicted: {predicted.item()}, Actual: {actual_class}')
show_image(X, f'Predicted: {predicted.item()}, Actual: {actual_class}')


# 绘制卷积层输出的特征图
def plot_feature_maps(feature_maps, layer_name, layer_idx, num_cols=8):
    num_feature_maps = feature_maps.shape[1]
    num_rows = (num_feature_maps + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    fig.suptitle(f'{layer_name} Layer {layer_idx}', fontsize=16)

    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_feature_maps:
                axes[i, j].imshow(feature_maps[0, idx].detach().cpu().numpy(), cmap='gray')
                axes[i, j].axis('off')
            else:
                axes[i, j].remove()
    plt.show()


# 输出每个卷积层的输出图像
n = 0
for (layer, layer_idx), output in outputs.items():
    n += 1
    if n < 10:
        print(f"Layer {layer_idx}: {layer} - Output shape: {output.shape}")
        plot_feature_maps(output, layer.__class__.__name__, layer_idx)
