# NOSA v3.1 Model Backbone
# Plain U-Net family design (no attention, no residual blocks), optimized for 3D:
# - Anisotropic stem kernels: (1,3,3) then (3,3,3)
# - Anisotropic early downsampling: (1,2,2), then isotropic (2,2,2)
# - Asymmetric capacity: stronger encoder, lighter decoder
# - InstanceNorm + LeakyReLU + Kaiming init for stable small-batch 3D training

import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic Conv-Norm-Activation block used throughout encoder/decoder stages
class ConvNormAct3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), dropout=0.0):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DoubleConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_1=(3, 3, 3),
        kernel_2=(3, 3, 3),
        dropout=0.0,
    ):
        super().__init__()
        self.conv1 = ConvNormAct3D(in_channels, out_channels, kernel_size=kernel_1, dropout=dropout)
        self.conv2 = ConvNormAct3D(out_channels, out_channels, kernel_size=kernel_2, dropout=0.0)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Downsample3D(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.down = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        return self.down(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, stride=(2, 2, 2), dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False)
        self.conv = DoubleConv3D(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    # 3D-optimized asymmetric U-Net backbone (no residual/attention modules).
    # Defaults target BraTS multi-modal training:
    # - input channels: 4 (t1, t1ce, t2, flair)
    # - output channels: 3 region heads (WT, TC, ET)

    def __init__(self, n_channels=4, n_classes=3, base_filters=32):
        super().__init__()

        c1 = base_filters
        c2 = base_filters * 2
        c3 = base_filters * 4
        c4 = base_filters * 8
        cb = base_filters * 12

        d1 = base_filters * 6
        d2 = base_filters * 4
        d3 = base_filters * 2
        d4 = base_filters

        # ENCODER (Contracting Path)
        self.enc1 = DoubleConv3D(
            n_channels,
            c1,
            kernel_1=(1, 3, 3),
            kernel_2=(3, 3, 3),
            dropout=0.05,
        )
        self.down1 = Downsample3D(c1, stride=(1, 2, 2))

        self.enc2 = DoubleConv3D(c1, c2, kernel_1=(3, 3, 3), kernel_2=(3, 3, 3), dropout=0.05)
        self.down2 = Downsample3D(c2, stride=(2, 2, 2))

        self.enc3 = DoubleConv3D(c2, c3, kernel_1=(3, 3, 3), kernel_2=(3, 3, 3), dropout=0.10)
        self.down3 = Downsample3D(c3, stride=(2, 2, 2))

        self.enc4 = DoubleConv3D(c3, c4, kernel_1=(3, 3, 3), kernel_2=(3, 3, 3), dropout=0.10)
        self.down4 = Downsample3D(c4, stride=(2, 2, 2))

        # BOTTLENECK (Deepest feature extraction)
        self.bottleneck = DoubleConv3D(
            c4,
            cb,
            kernel_1=(3, 3, 3),
            kernel_2=(3, 3, 3),
            dropout=0.15,
        )

        # DECODER (Expanding Path)
        self.up1 = UpBlock3D(cb, c4, d1, stride=(2, 2, 2), dropout=0.10)
        self.up2 = UpBlock3D(d1, c3, d2, stride=(2, 2, 2), dropout=0.10)
        self.up3 = UpBlock3D(d2, c2, d3, stride=(2, 2, 2), dropout=0.05)
        self.up4 = UpBlock3D(d3, c1, d4, stride=(1, 2, 2), dropout=0.05)

        # OUTPUT HEAD
        self.out_conv = nn.Conv3d(d4, n_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x):
        # Encoder pathway
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        # Bottleneck pathway
        b = self.bottleneck(self.down4(e4))

        # Decoder pathway with skip connections
        y = self.up1(b, e4)
        y = self.up2(y, e3)
        y = self.up3(y, e2)
        y = self.up4(y, e1)

        return self.out_conv(y)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.InstanceNorm3d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    model = UNet3D(n_channels=4, n_classes=3, base_filters=32)
    x = torch.randn(1, 4, 128, 128, 128)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
