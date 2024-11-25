import os

# 创建数据文件并写入数据样本
os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 创建文件夹时，如果文件夹已经存在，则忽略错误
data_file = os.path.join('..', 'data', 'house_tiny.csv')  # 定义数据文件路径
with open(data_file, 'w') as f:  # 打开数据文件
    f.write("NumRooms,Alley,Price\n")  # 写入列名
    f.write("NA,Pave,127500\n")  # 每行表示一个数据样本
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")

# 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data, type(data), sep='\n')
#    NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000
# <class 'pandas.core.frame.DataFrame'>

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  # 输入和输出分别是NumRooms和Alley，Price
inputs = inputs.fillna(inputs.mean())  # 用平均值填充缺失值
print(inputs, type(inputs), sep='\n')
#    NumRooms Alley
# 0       3.0  Pave
# 1       2.0   NaN
# 2       4.0   NaN
# 3       3.0   NaN
# <class 'pandas.core.frame.DataFrame'>

inputs = pd.get_dummies(inputs, dummy_na=True)  # 独热编码
# pd.get_dummies(inputs, dummy_na=True) 会对输入的 inputs DataFrame 进行独热编码，并将结果存储在新的DataFrame中。
# dummy_na=True可以指定是否为缺失值创建指示变量，如果为True，则会为缺失值单独创建一列。
print(inputs)
#    NumRooms  Alley_Pave  Alley_nan
# 0       3.0           1          0
# 1       2.0           0          1
# 2       4.0           0          1
# 3       3.0           0          1

import torch

X = torch.tensor(inputs.to_numpy(dtype=float))  # 转换为张量 Tensor, to_numpy() 方法可以将 DataFrame 转换为 numpy 数组
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, y, sep='\n')
# tensor([[3., 1., 0.],
#         [2., 0., 1.],
#         [4., 0., 1.],
#         [3., 0., 1.]], dtype=torch.float64)
# tensor([127500., 106000., 178100., 140000.], dtype=torch.float64)

'''
作业：
- 删除缺失值最多的列。
- 将预处理后的数据集转换为张量格式。
'''
missing_counts = data.isnull().sum()  # 统计缺失值个数
max_missing_col = missing_counts.idxmax()  # 找出拥有最大缺失值个数的列名
data.drop(columns=max_missing_col, inplace=True)  # 删除拥有最大缺失值个数的列
print(data, type(data), sep='\n')
#    NumRooms   Price
# 0       NaN  127500
# 1       2.0  106000
# 2       4.0  178100
# 3       NaN  140000
# <class 'pandas.core.frame.DataFrame'>
data_new = torch.tensor(data.to_numpy(dtype=float))  # 转换为张量 Tensor
print(data_new, type(data_new), sep='\n')
# tensor([[       nan, 1.2750e+05],
#         [2.0000e+00, 1.0600e+05],
#         [4.0000e+00, 1.7810e+05],
#         [       nan, 1.4000e+05]], dtype=torch.float64)
# <class 'torch.Tensor'>









