# anUNet v3.2.1 model backbone
# Residualized U-Net design (no attention), optimized for 3D:
# - Anisotropic stem kernels: (1,3,3) then (3,3,3)
# - Anisotropic early downsampling: (1,2,2), then isotropic (2,2,2)
# - Asymmetric capacity: stronger encoder, lighter decoder
# - Pre-activation residual blocks + InstanceNorm + LeakyReLU + Kaiming init

import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual block: pre-activation conv path + identity/1x1 shortcut
class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_1=(3, 3, 3),
        kernel_2=(3, 3, 3),
        dropout=0.1,
    ):
        super().__init__()
        pad1 = tuple(k // 2 for k in kernel_1)
        pad2 = tuple(k // 2 for k in kernel_2)

        self.norm1 = nn.InstanceNorm3d(in_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_1, padding=pad1, bias=False)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_2, padding=pad2, bias=False)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        y = self.conv1(self.act1(self.norm1(x)))
        y = self.drop(y)
        y = self.conv2(self.act2(self.norm2(y)))
        return y + residual


class ResidualStage3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=2,
        first_kernel_1=(3, 3, 3),
        first_kernel_2=(3, 3, 3),
        dropout=0.0,
    ):
        super().__init__()
        blocks = [
            ResidualBlock3D(
                in_channels,
                out_channels,
                kernel_1=first_kernel_1,
                kernel_2=first_kernel_2,
                dropout=dropout,
            )
        ]

        for _ in range(max(0, num_blocks - 1)):
            blocks.append(
                ResidualBlock3D(
                    out_channels,
                    out_channels,
                    kernel_1=(3, 3, 3),
                    kernel_2=(3, 3, 3),
                    dropout=0.0,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Downsample3D(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.down = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        return self.down(x)


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        stride=(2, 2, 2),
        dropout=0.0,
        num_blocks=2,
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False)
        self.conv = ResidualStage3D(
            out_channels + skip_channels,
            out_channels,
            num_blocks=num_blocks,
            dropout=dropout,
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    # 3D residual U-Net with deeper stages and decoder features for deep supervision.

    def __init__(
        self,
        n_channels=4,
        n_classes=3,
        base_filters=32,
        stage_blocks=3,
        bottleneck_blocks=4,
    ):
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
        self.d2 = d2
        self.d3 = d3

        # ENCODER (Contracting Path)
        self.enc1 = ResidualStage3D(
            n_channels,
            c1,
            num_blocks=stage_blocks,
            first_kernel_1=(1, 3, 3),
            first_kernel_2=(3, 3, 3),
            dropout=0.03,
        )
        self.down1 = Downsample3D(c1, stride=(1, 2, 2))

        self.enc2 = ResidualStage3D(c1, c2, num_blocks=stage_blocks, dropout=0.03)
        self.down2 = Downsample3D(c2, stride=(2, 2, 2))

        self.enc3 = ResidualStage3D(c2, c3, num_blocks=stage_blocks, dropout=0.07)
        self.down3 = Downsample3D(c3, stride=(2, 2, 2))

        self.enc4 = ResidualStage3D(c3, c4, num_blocks=stage_blocks, dropout=0.07)
        self.down4 = Downsample3D(c4, stride=(2, 2, 2))

        # BOTTLENECK (Deepest residual stage)
        self.bottleneck = ResidualStage3D(
            c4,
            cb,
            num_blocks=bottleneck_blocks,
            dropout=0.10,
        )

        # DECODER (Expanding Path)
        self.up1 = UpBlock3D(cb, c4, d1, stride=(2, 2, 2), dropout=0.07, num_blocks=stage_blocks)
        self.up2 = UpBlock3D(d1, c3, d2, stride=(2, 2, 2), dropout=0.07, num_blocks=stage_blocks)
        self.up3 = UpBlock3D(d2, c2, d3, stride=(2, 2, 2), dropout=0.03, num_blocks=stage_blocks)
        self.up4 = UpBlock3D(d3, c1, d4, stride=(1, 2, 2), dropout=0.03, num_blocks=stage_blocks)

        # OUTPUT HEAD
        self.out_conv = nn.Conv3d(d4, n_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x, return_decoder_features=False):
        # Encoder pathway
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        # Bottleneck pathway
        b = self.bottleneck(self.down4(e4))

        # Decoder pathway with skip connections
        y1 = self.up1(b, e4)
        y2 = self.up2(y1, e3)
        y3 = self.up3(y2, e2)
        y4 = self.up4(y3, e1)
        out = self.out_conv(y4)
        if return_decoder_features:
            return out, y2, y3

        return out

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
