import torch as th
import torch.nn as nn
import numpy as np


class AttentionBlock(nn.Module):
    def __init__(self, in_channel):
        """
        self-attention mechanism block
        处理前后形状相同
        """
        super().__init__()
        self.in_channel = in_channel
        # self.norm = nn.BatchNorm2d(in_channel)
        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 3, kernel_size=1)
        self.out = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        Q, K, V = th.split(qkv, self.in_channel, dim=1)
        QK_matmul = th.bmm(Q.reshape(b, h*w, c), K.reshape(b, c, h*w))
        attn = th.softmax(QK_matmul / np.sqrt(c), dim=-1)
        out = th.bmm(attn, V.reshape(b, h*w, c)).reshape(b, h, w, c).permute(0, 3, 1, 2)
        out = self.out(out)
        return out + x



