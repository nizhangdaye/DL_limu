import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)
# tensor([0, 1, 2, 3])

y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print(x2, y2)
# tensor([0, 1, 2, 3]) tensor([0., 0., 0., 0.])

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)


# {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}


# 加载和保存模型参数
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
