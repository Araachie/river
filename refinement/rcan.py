import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, act=nn.ReLU(True)):
        m = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), stride=stride, bias=bias)]
        if act is not None:
            m.append(act)
        super().__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, act=nn.ReLU(True)):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x) + x


class CALayer(nn.Module):  # Channel Attention (CA) Layer
    def __init__(self, channel, reduction=16):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):  # Residual Channel Attention Block
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, act=nn.ReLU(True)):
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0:
                modules_body.append(act)
        if reduction > 0:
            modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x) + x


class ResidualGroup(nn.Module):  # Residual Group (RG)
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super().__init__()
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, act=nn.ReLU(True)) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x) + x


class RCAN(nn.Module):  # Residual Channel Attention Network
    def __init__(self, n_resgroups=4, n_resblocks=5, n_feats=64, reduction=8,
                 tanh=False, kernel_size=3, conv=default_conv):
        super().__init__()
        self.head = conv(6, n_feats, kernel_size)
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)
        if tanh:
            self.tail = nn.Sequential(conv(n_feats, 3, kernel_size), nn.Tanh())
        else:
            self.tail = conv(n_feats, 3, kernel_size)

    def forward(self, x):
        x = self.head(x)
        return self.tail(self.body(x) + x)
