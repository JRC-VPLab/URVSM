import torch
import torch.nn as nn
import torch.nn.functional as F


'''=============================== UNet ==============================='''

class ConvBlock(nn.Module):
    def __init__(self, inch, outch, stride=1, dilation=1):
        # Residual Block
        # inch: input feature channel
        # outch: output feature channel
        # stride: the stride of  convolution layer
        super(ConvBlock, self ).__init__()
        assert(stride == 1 or stride == 2)

        self.conv1 = nn.Conv2d(inch, outch, 3, stride, padding=dilation, bias=False,
                dilation = dilation)
        self.bn1 = nn.BatchNorm2d(outch)
        self.conv2 = nn.Conv2d(outch, outch, 3, 1, padding=dilation, bias=False,
                dilation = dilation)
        self.bn2 = nn.BatchNorm2d(outch)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        return out


class UNet_encoder(nn.Module):
    def __init__(self, depth, in_channel=3):
        super().__init__()
        self.depth = depth

        self.e_0 = ConvBlock(in_channel, self.depth[0], 1)
        self.e_1 = ConvBlock(self.depth[0], self.depth[1], 2)
        self.e_2 = ConvBlock(self.depth[1], self.depth[2], 2)
        self.e_3 = ConvBlock(self.depth[2], self.depth[3], 2)

    def forward(self, x):
        x1 = self.e_0(x)
        x2 = self.e_1(x1)
        x3 = self.e_2(x2)
        x4 = self.e_3(x3)

        return x1, x2, x3, x4


class UNet_decoder(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth

        self.d_2 = ConvBlock(self.depth[3] + self.depth[2], self.depth[1], 1)
        self.d_1 = ConvBlock(self.depth[1] + self.depth[1], self.depth[0], 1)
        self.d_0 = ConvBlock(self.depth[0], 3, 1)

        self.conv_out = nn.Conv2d(3, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2, x3, x4):
        _, _, nh, nw = x3.size()
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear')
        y1 = self.d_2(torch.cat([x3, x4], dim=1))

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y2 = self.d_1(torch.cat([y1, x2], dim=1))

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear')
        y3 = self.d_0(y2)

        y3 = self.conv_out(y3)
        pred = self.sigmoid(y3)

        return pred


class UNet_vanilla(nn.Module):
    def __init__(self, depth=[32, 64, 128, 256]):
        super().__init__()

        self.encoder = UNet_encoder(depth, in_channel=4)
        self.decoder = UNet_decoder(depth)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        y = self.decoder(x1, x2, x3, x4)

        return y



'''=============================== ResDoUNet ==============================='''

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 4), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 4), channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


class DOConv2d(nn.Module):
    """
    Depthwise Over-parameterized Conv (practical proxy):
    - Standard 3x3 conv in parallel with a depthwise 3x3 conv (groups=in_ch).
    - Outputs are summed; BN+ReLU are applied by the caller.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=bias)
        self.dw_to_out = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else None

    def forward(self, x):
        y1 = self.pw(x)
        y2 = self.dw(x)
        if self.dw_to_out is not None:
            y2 = self.dw_to_out(y2)
        return y1 + y2


class ResDOBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = DOConv2d(in_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = DOConv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class PFB(nn.Module):
    """Pooling Fusion Block: avg-pool and max-pool in parallel, concat, 1x1 squeeze."""
    def __init__(self, channels: int):
        super().__init__()
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.squeeze = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        a = self.avg(x)
        m = self.max(x)
        y = torch.cat([a, m], dim=1)
        y = self.act(self.bn(self.squeeze(y)))
        return y


class AFB(nn.Module):
    """
    Attention Fusion Block for skip connection:
    Inputs: low-level (encoder) and high-level (decoder or bottleneck).
    We **do not** rely on fixed 2× upsampling anywhere. Instead, we
    always resize `high` to exactly match the spatial size of `low`.
    Branches use dilations 1, 2, 4. SE attention + 1x1 fuse.
    """
    def __init__(self, low_ch: int, high_ch: int, out_ch: int):
        super().__init__()
        in_ch = low_ch + high_ch
        mid = max(out_ch // 2, 32)

        def branch(d):
            return nn.Sequential(
                nn.Conv2d(in_ch, mid, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )

        self.b1 = branch(1)
        self.b2 = branch(2)
        self.b3 = branch(4)
        self.se = SEBlock(3 * mid)
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * mid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, low, high):
        # Resize high to match low spatially; no fixed scale factors used
        if low.shape[-2:] != high.shape[-2:]:
            high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([low, high], dim=1)
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        y = self.se(y)
        y = self.fuse(y)
        return y


# ----------------------------
# Network (3-level encoder-decoder; lighter; base_ch=16)
# ----------------------------

class ResDO_UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=16):
        super().__init__()
        # 3 encoder stages instead of 4; smaller base width
        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8  # c4 is bottleneck width

        # Encoder
        self.enc1 = ResDOBlock(in_ch, c1)
        self.pfb1 = PFB(c1)

        self.enc2 = ResDOBlock(c1, c2)
        self.pfb2 = PFB(c2)

        self.enc3 = ResDOBlock(c2, c3)
        self.pfb3 = PFB(c3)

        # Bottleneck (no extra pooling beyond pfb3)
        self.bott = ResDOBlock(c3, c4)

        # Decoder (NO nn.Upsample layers; AFB handles exact-size matching)
        self.afb3 = AFB(low_ch=c3, high_ch=c4, out_ch=c3)
        self.dec3 = ResDOBlock(c3, c3)

        self.afb2 = AFB(low_ch=c2, high_ch=c3, out_ch=c2)
        self.dec2 = ResDOBlock(c2, c2)

        self.afb1 = AFB(low_ch=c1, high_ch=c2, out_ch=c1)
        self.dec1 = ResDOBlock(c1, c1)

        # Head
        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x);  d1 = self.pfb1(e1)
        e2 = self.enc2(d1); d2 = self.pfb2(e2)
        e3 = self.enc3(d2); d3 = self.pfb3(e3)

        # Bottleneck
        b = self.bott(d3)

        # Decoder (AFB resizes high feature to match low; robust to arbitrary HxW)
        x3 = self.afb3(e3, b)
        x3 = self.dec3(x3)

        x2 = self.afb2(e2, x3)
        x2 = self.dec2(x2)

        x1 = self.afb1(e1, x2)
        x1 = self.dec1(x1)

        out = self.head(x1)
        out = self.sigmoid(out)
        return out