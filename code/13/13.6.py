import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

# @save
d2l.DATA_HUB['banana-detection'] = (  # 下载数据集
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')


# @save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')  # 下载解压数据集
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')  # 标签文件路径
    csv_data = pd.read_csv(csv_fname)  # 读取标签文件
    csv_data = csv_data.set_index('img_name')  # 设置索引
    images, targets = [], []
    # 把图片、标号全部读到内存里面
    for img_name, target in csv_data.iterrows():  # 遍历数据集中的图像和标签
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256  # 标签值归一化到0-1之间


# @save
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""

    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features))
              + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


# @save
def load_data_bananas(batch_size):  # 定义一个函数来小批量加载数据集
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter


batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))  # batch 是一个 tuple (features, labels)
print(batch[0].shape, batch[1].shape)
# torch.Size([32, 3, 256, 256]) torch.Size([32, 1, 5])

imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255  # permute 将通道维移到最后面 并将像素值归一化到0-1之间
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
