import torch
import torch.nn as nn
import numpy as np
B = 1
data = (256*torch.rand((B,1,64,64))).round_()

hist_nondiff = torch.histc(data, bins=256, min=0, max=256)

print(hist_nondiff)

class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

    def forward(self, x):
        ## input should be reshaped into [B, len]
        x = x.view(B, -1)
        x = torch.unsqueeze(x, 1) - torch.unsqueeze(self.centers, -1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=-1)
        return x

    def forward_cmp(self, input):
        ## input should be reshaped into [B, len]
        b, c, h, w = input.shape
        input = input.view(B, -1)
        x = torch.unsqueeze(input, 1) - torch.unsqueeze(self.centers, -1)
        x = torch.sigmoid(self.sigma * x)
        diff = torch.cat([torch.ones((b,1,h*w),device=input.device), x],dim=1) - torch.cat([x, torch.zeros((b,1,h*w),device=input.device)],dim=1)

        diff = diff.sum(dim=-1)
        diff[:,-2] += diff[:,-1]
        return diff[:,:-1]

softhist = SoftHistogram(bins=256, min=0, max=256, sigma=100)

data.requires_grad = True
hist = softhist.forward_cmp(data)
print(hist)
loss = nn.L1Loss()
print(loss(hist.squeeze(),hist_nondiff))
# print(data.grad.max())