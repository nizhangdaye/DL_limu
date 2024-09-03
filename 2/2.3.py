import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x - y, x / y, x ** y, sep='\n')
# tensor(5.)
# tensor(1.)
# tensor(1.5000)
# tensor(9.)

x = torch.arange(4)
print(x)
# tensor([0, 1, 2, 3])

print(x[3])
# tensor(3)

print(x.shape)
# torch.Size([4])

A = torch.arange(20).reshape(5, 4)
print(A)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15],
#         [16, 17, 18, 19]])

print(A.T)
# tensor([[ 0,  4,  8, 12, 16],
#         [ 1,  5,  9, 13, 17],
#         [ 2,  6, 10, 14, 18],
#         [ 3,  7, 11, 15, 19]])

X = torch.arange(24).reshape(2, 3, 4)
print(X)
# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将 A 的一个副本分配给 B
print(A, A + B, sep='\n')
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.],
#         [16., 17., 18., 19.]])
# tensor([[ 0.,  2.,  4.,  6.],
#         [ 8., 10., 12., 14.],
#         [16., 18., 20., 22.],
#         [24., 26., 28., 30.],
#         [32., 34., 36., 38.]])

print(A * B)
# tensor([[  0.,   1.,   4.,   9.],
#         [ 16.,  25.,  36.,  49.],
#         [ 64.,  81., 100., 121.],
#         [144., 169., 196., 225.],
#         [256., 289., 324., 361.]])

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape, sep='\n')
# tensor([[[ 2,  3,  4,  5],
#          [ 6,  7,  8,  9],
#          [10, 11, 12, 13]],
#
#         [[14, 15, 16, 17],
#          [18, 19, 20, 21],
#          [22, 23, 24, 25]]])
# torch.Size([2, 3, 4])

x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())
# tensor([0., 1., 2., 3.]) tensor(6.)

print(A.shape, A.sum())
# torch.Size([5, 4]) tensor(190.)

A_sum_axis0 = A.sum(axis=0)  # 按行求和
print(A_sum_axis0, A_sum_axis0.shape)
# tensor([40., 45., 50., 55.]) torch.Size([4])

A_sum_axis1 = A.sum(axis=1)  # 按列求和
print(A_sum_axis1, A_sum_axis1.shape)
# tensor([ 6., 22., 38., 54., 70.]) torch.Size([5])

print(A.sum(axis=[0, 1]))  # 结果和A.sum()相同
# tensor(190.)

print(A.mean(axis=0), A.sum(axis=0)/A.shape[0])  # 按行求均值
# tensor([ 2.,  5.,  8., 11.]) tensor([ 2.,  5.,  8., 11.])

sum_A = A.sum(axis=1, keepdim=True)  # 按列求和，保持维度
print(sum_A, sum_A.shape, sep='\n')
# tensor([[ 6.],
#         [22.],
#         [38.],
#         [54.],
#         [70.]])
# torch.Size([5, 1])

print(A/sum_A)  # 广播机制，按列除以列和
# tensor([[0.0000, 0.1667, 0.3333, 0.5000],
#         [0.1818, 0.2273, 0.2727, 0.3182],
#         [0.2105, 0.2368, 0.2632, 0.2895],
#         [0.2222, 0.2407, 0.2593, 0.2778],
#         [0.2286, 0.2429, 0.2571, 0.2714]])

print(A.cumsum(axis=0))  # 按行累积求和
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  6.,  8., 10.],
#         [12., 15., 18., 21.],
#         [24., 28., 32., 36.],
#         [40., 45., 50., 55.]])

y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))  # 点乘，内积 x.T * y
# tensor([0., 1., 2., 3.]) tensor([1., 1., 1., 1.]) tensor(6.)

print(torch.sum(x * y))
# tensor(6.)

print(A.shape, x.shape, torch.mv(A, x))  # 矩阵向量乘法
# torch.Size([5, 4]) torch.Size([4]) tensor([ 14.,  38.,  62.,  86., 110.])

B = torch.ones(4, 3)
print(torch.mm(A, B))  # 矩阵乘法
# tensor([[ 6.,  6.,  6.],
#         [22., 22., 22.],
#         [38., 38., 38.],
#         [54., 54., 54.],
#         [70., 70., 70.]])

u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
# tensor(5.)

print(torch.abs(u).sum())
# tensor(7.)

print(torch.norm(torch.ones((4, 9))))  # Frobenius范数
# tensor(6.)

