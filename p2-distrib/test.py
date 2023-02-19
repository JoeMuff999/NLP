import torch

x = torch.randn(2,3,27)
print(x.shape)

y = x.view(-1, 27)
print(y.shape)