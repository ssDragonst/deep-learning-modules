import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvtf


class TwoConvGroup(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=0, activate=F.relu):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding)
        self.activate = activate

    def forward(self, x):
        out = self.activate(self.conv1(x))
        out = self.activate(self.conv2(out))
        return out


class UpSampleStep(nn.Module):
    def __init__(self, in_channel, out_channel, up_kernel_size=2, conv_kernel_size=3, padding=0, activate=F.relu):
        """
        卷积尺寸计算：Out = ((input + 2padding -kernel) / stride) + 1
        反卷积尺寸计算：Out = stride(input - 1) - 2padding + kernel

        :param in_channel: 反卷积前的channel，反卷积后channel减半，拼接后恢复
        """
        super().__init__()
        self.activate = activate
        self.deconv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=up_kernel_size, stride=2, padding=padding)
        self.conv = TwoConvGroup(in_channel, out_channel, conv_kernel_size, padding, activate)

    def forward(self, x, concat_data):
        up = self.deconv(x)
        assert concat_data.shape == up.shape, '两个要拼接的Tensor形状不一致'
        out = self.conv(th.cat([up, concat_data], dim=1))
        return out


# 一个用上面个两个基本结构组建U-Net的demo
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = TwoConvGroup(1, 64)
        self.down2 = TwoConvGroup(64, 128)
        self.down3 = TwoConvGroup(128, 256)
        self.down4 = TwoConvGroup(256, 512)
        self.down5 = TwoConvGroup(512, 1024)
        self.up1 = UpSampleStep(1024, 512)
        self.up2 = UpSampleStep(512, 256)
        self.up3 = UpSampleStep(256, 128)
        self.up4 = UpSampleStep(128, 64)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(th.max_pool2d(c1, 2))
        c3 = self.down3(th.max_pool2d(c2, 2))
        c4 = self.down4(th.max_pool2d(c3, 2))
        c5 = self.down5(th.max_pool2d(c4, 2))
        u1 = self.up1(c5, tvtf.crop(c4, 4, 4, 56, 56))
        u2 = self.up2(u1, tvtf.crop(c3, 16, 16, 104, 104))
        u3 = self.up3(u2, tvtf.crop(c2, 40, 40, 200, 200))
        u4 = self.up4(u3, tvtf.crop(c1, 88, 88, 392, 392))
        return u4


if __name__ == '__main__':
    unet = UNet()
    x = th.randint(3, (1, 1, 572, 572))
    x = th.tensor(x, dtype=th.float).to('cuda')
    unet.to('cuda')
    res = unet(x)
    print(res.shape)

        

