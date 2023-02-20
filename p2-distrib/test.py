import torch

x = torch.randn(2,3,27)
print(x.shape)

y = x.view(-1, 27)
print(y.shape)

a = torch.randn(20, 64)
b = torch.randn(20, 64)

cat = torch.cat((a, b), dim=0)
print(cat.shape)