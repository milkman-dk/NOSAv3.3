# NOSA v3.3 Model Backbone
# Residualized 3D U-Net with multi-scale attention:
# - Modality attention on the 4-channel input stem
# - CAM/SE recalibration at enc4 and bottleneck
# - Axial attention at enc4 for medium-resolution long-range context
# - PAM at the bottleneck for global spatial context
# - Attention gates on the e2/e3/e4 skip connections

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    # Squeeze-and-excitation block: channel attention via global pooling â†’ FC â†’ scaling
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction) #reduction_ratio
        self.pool = nn.AdaptiveAvgPool3d(1) #global_context
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True) #bottleneck_fc
        self.act = nn.ReLU(inplace=True) #activation
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True) #expand_fc
        self.gate = nn.Sigmoid() #channel_gate

    def forward(self, x):
        scale = self.pool(x) #squeeze
        scale = self.fc2(self.act(self.fc1(scale))) #excitation
        return x * self.gate(scale) #channel_scaling


class ModalityAttention3D(SEBlock3D):
    def __init__(self, channels):
        super().__init__(channels, reduction=2)


class AttentionGate3D(nn.Module):
    # Gated skip connection: combines skip signal + gating signal via spatial attention
    def __init__(self, skip_channels, gating_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or max(1, min(skip_channels, gating_channels) // 2) #channel_reduction
        self.theta = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False) #skip_projection
        self.phi = nn.Conv3d(gating_channels, inter_channels, kernel_size=1, bias=False) #gate_projection
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True) #activation
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True) #attention_map
        self.gate = nn.Sigmoid() #sigmoid_gate

    def forward(self, skip, gating):
        attn = self.theta(skip) + self.phi(gating) #combine_signals
        attn = self.gate(self.psi(self.act(attn))) #attention_weights
        return skip * attn #gated_skip


class AxialAttention3D(nn.Module):
    # Multi-headed self-attention applied separately on depth, height, width axes â†’ long-range context
    def __init__(self, channels, num_heads=8, dropout=0.0):
        super().__init__()
        safe_heads = min(num_heads, channels) #head_channel_divisibility
        while safe_heads > 1 and channels % safe_heads != 0:
            safe_heads -= 1 #ensure_divisibility
        self.num_heads = safe_heads #num_heads

        self.depth_norm = nn.LayerNorm(channels) #layernorm_depth
        self.height_norm = nn.LayerNorm(channels) #layernorm_height
        self.width_norm = nn.LayerNorm(channels) #layernorm_width
        self.depth_attn = nn.MultiheadAttention(channels, self.num_heads, dropout=dropout, batch_first=True) #mha_depth
        self.height_attn = nn.MultiheadAttention(channels, self.num_heads, dropout=dropout, batch_first=True) #mha_height
        self.width_attn = nn.MultiheadAttention(channels, self.num_heads, dropout=dropout, batch_first=True) #mha_width
        self.gamma = nn.Parameter(torch.zeros(1)) #residual_scaling

    def _apply_attention(self, seq, attn, norm):
        orig_dtype = seq.dtype #dtype_preservation
        with torch.autocast(device_type=seq.device.type, enabled=False):
            seq_fp32 = seq.float() #fp32_casting
            seq_norm = norm(seq_fp32) #normalize
            attn_out, _ = attn(seq_norm, seq_norm, seq_norm, need_weights=False) #self_attention
        return seq + self.gamma.to(dtype=orig_dtype) * attn_out.to(dtype=orig_dtype) #residual_add

    def forward(self, x):
        b, c, d, h, w = x.shape #unpack_shape

        seq_d = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, c) #sequence_depth
        seq_d = self._apply_attention(seq_d, self.depth_attn, self.depth_norm) #attn_depth
        x = seq_d.reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2) #reshape_after_depth

        seq_h = x.permute(0, 2, 4, 3, 1).reshape(b * d * w, h, c) #sequence_height
        seq_h = self._apply_attention(seq_h, self.height_attn, self.height_norm) #attn_height
        x = seq_h.reshape(b, d, w, h, c).permute(0, 4, 1, 3, 2) #reshape_after_height

        seq_w = x.permute(0, 2, 3, 4, 1).reshape(b * d * h, w, c) #sequence_width
        seq_w = self._apply_attention(seq_w, self.width_attn, self.width_norm) #attn_width
        return seq_w.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3) #reshape_after_width


class PositionAttention3D(nn.Module):
    # Spatial attention map: QÂ·K^T softmax â†’ weights global positions, scaled by gamma
    def __init__(self, channels):
        super().__init__()
        reduced = max(1, channels // 8) #reduction_factor
        self.query = nn.Conv3d(channels, reduced, kernel_size=1, bias=False) #query_projection
        self.key = nn.Conv3d(channels, reduced, kernel_size=1, bias=False) #key_projection
        self.value = nn.Conv3d(channels, channels, kernel_size=1, bias=False) #value_projection
        self.gamma = nn.Parameter(torch.zeros(1)) #residual_scaling

    def forward(self, x):
        b, c, d, h, w = x.shape #unpack_shape
        n = d * h * w #flatten_spatial
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float() #fp32_casting
            query = self.query(x_fp32).reshape(b, -1, n).transpose(1, 2) #query_matrix
            key = self.key(x_fp32).reshape(b, -1, n) #key_matrix
            scale = query.shape[-1] ** -0.5 #scaling_factor
            attn = torch.softmax(torch.bmm(query, key) * scale, dim=-1) #attention_weights
            value = self.value(x_fp32).reshape(b, c, n) #value_matrix
            out = torch.bmm(value, attn.transpose(1, 2)).reshape(b, c, d, h, w) #spatial_attention_out
        return x + self.gamma.to(dtype=x.dtype) * out.to(dtype=x.dtype) #residual_add


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
        pad1 = tuple(k // 2 for k in kernel_1) #padding
        pad2 = tuple(k // 2 for k in kernel_2)

        self.norm1 = nn.InstanceNorm3d(in_channels, affine=True) #normalization
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True) #activation
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_1, padding=pad1, bias=False) #convolution
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity() #dropout

        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True) #normalization_2
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True) #activation_2
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_2, padding=pad2, bias=False) #convolution_2

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False) #projection_shortcut
        else:
            self.shortcut = nn.Identity() #identity_shortcut

    def forward(self, x):
        residual = self.shortcut(x)
        y = self.conv1(self.act1(self.norm1(x)))
        y = self.drop(y)
        y = self.conv2(self.act2(self.norm2(y)))
        return y + residual


class ResidualStage3D(nn.Module):
    # Stack of residual blocks: first with custom kernels, rest (3,3,3), all with optional dropout
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
            ResidualBlock3D( #first_block
                in_channels,
                out_channels,
                kernel_1=first_kernel_1,
                kernel_2=first_kernel_2,
                dropout=dropout,
            )
        ]

        for _ in range(max(0, num_blocks - 1)):
            blocks.append( #additional_blocks
                ResidualBlock3D(
                    out_channels,
                    out_channels,
                    kernel_1=(3, 3, 3),
                    kernel_2=(3, 3, 3),
                    dropout=0.0,
                )
            )

        self.blocks = nn.Sequential(*blocks) #stage_sequence

    def forward(self, x):
        return self.blocks(x) #forward_pass


class Downsample3D(nn.Module):
    # Strided convolution for 2x or 4x spatial reduction in encoder
    def __init__(self, channels, stride):
        super().__init__()
        self.down = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False) #strided_conv

    def forward(self, x):
        return self.down(x) #downsample


class UpBlock3D(nn.Module):
    # Decoder block: transpose conv upsample + attention gate + concatenate skip + residual stage
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        stride=(2, 2, 2),
        dropout=0.0,
        num_blocks=2,
        use_attention_gate=False,
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False) #transpose_conv
        self.gate = AttentionGate3D(skip_channels, out_channels) if use_attention_gate else None #attention_gate_optional
        self.conv = ResidualStage3D( #residual_stage
            out_channels + skip_channels,
            out_channels,
            num_blocks=num_blocks,
            dropout=dropout,
        )

    def forward(self, x, skip):
        x = self.up(x) #upsample
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False) #interpolate_if_needed
        if self.gate is not None:
            skip = self.gate(skip, x) #apply_attention_gate
        x = torch.cat([x, skip], dim=1) #concatenate_skip
        return self.conv(x) #residual_processing


class UNet3D(nn.Module):
    # 3D U-Net backbone: encoder â†’ bottleneck â†’ decoder with multi-scale attention (SE, PAM, Axial, AttentionGates)
    
    def __init__(
        self,
        n_channels=4,
        n_classes=3,
        base_filters=32,
        stage_blocks=3,
        bottleneck_blocks=4,
    ):
        super().__init__()

        c1 = base_filters #encoder_ch1
        c2 = base_filters * 2 #encoder_ch2
        c3 = base_filters * 4 #encoder_ch3
        c4 = base_filters * 8 #encoder_ch4
        cb = base_filters * 12 #bottleneck_ch

        d1 = base_filters * 6 #decoder_ch1
        d2 = base_filters * 4 #decoder_ch2
        d3 = base_filters * 2 #decoder_ch3
        d4 = base_filters #decoder_ch4
        self.d2 = d2 #save_for_export
        self.d3 = d3 #save_for_export
        self.modality_attention = ModalityAttention3D(n_channels) #input_attention

        self.enc1 = ResidualStage3D( #encoder_stage1
            n_channels,
            c1,
            num_blocks=stage_blocks,
            first_kernel_1=(1, 3, 3),
            first_kernel_2=(3, 3, 3),
            dropout=0.03,
        )
        self.down1 = Downsample3D(c1, stride=(1, 2, 2)) #downsample1

        self.enc2 = ResidualStage3D(c1, c2, num_blocks=stage_blocks, dropout=0.03) #encoder_stage2
        self.down2 = Downsample3D(c2, stride=(2, 2, 2)) #downsample2

        self.enc3 = ResidualStage3D(c2, c3, num_blocks=stage_blocks, dropout=0.07) #encoder_stage3
        self.down3 = Downsample3D(c3, stride=(2, 2, 2)) #downsample3

        self.enc4 = ResidualStage3D(c3, c4, num_blocks=stage_blocks, dropout=0.07) #encoder_stage4
        self.enc4_cam = SEBlock3D(c4, reduction=8) #encoder4_attention
        self.enc4_axial = AxialAttention3D(c4, num_heads=8) #encoder4_axial_attention
        self.down4 = Downsample3D(c4, stride=(2, 2, 2)) #downsample4

        self.bottleneck = ResidualStage3D( #bottleneck_stage
            c4,
            cb,
            num_blocks=bottleneck_blocks,
            dropout=0.10,
        )
        self.bottleneck_cam = SEBlock3D(cb, reduction=8) #bottleneck_channel_attention
        self.bottleneck_pam = PositionAttention3D(cb) #bottleneck_spatial_attention

        self.up1 = UpBlock3D(cb, c4, d1, stride=(2, 2, 2), dropout=0.07, num_blocks=stage_blocks, use_attention_gate=True) #decoder1
        self.up2 = UpBlock3D(d1, c3, d2, stride=(2, 2, 2), dropout=0.07, num_blocks=stage_blocks, use_attention_gate=True) #decoder2
        self.up3 = UpBlock3D(d2, c2, d3, stride=(2, 2, 2), dropout=0.03, num_blocks=stage_blocks, use_attention_gate=True) #decoder3
        self.up4 = UpBlock3D(d3, c1, d4, stride=(1, 2, 2), dropout=0.03, num_blocks=stage_blocks, use_attention_gate=False) #decoder4

        self.out_conv = nn.Conv3d(d4, n_classes, kernel_size=1) #output_projection

        self._init_weights() #weight_initialization

    def forward(self, x, return_decoder_features=False):
        x = self.modality_attention(x) #modality_attention
        e1 = self.enc1(x) #encode1
        e2 = self.enc2(self.down1(e1)) #encode2
        e3 = self.enc3(self.down2(e2)) #encode3
        e4 = self.enc4(self.down3(e3)) #encode4
        e4 = self.enc4_cam(e4) #enc4_channel_attention
        e4 = self.enc4_axial(e4) #enc4_axial_attention

        b = self.bottleneck(self.down4(e4)) #bottleneck
        b = self.bottleneck_cam(b) #bottleneck_channel_attention
        b = self.bottleneck_pam(b) #bottleneck_spatial_attention

        y1 = self.up1(b, e4) #decode1
        y2 = self.up2(y1, e3) #decode2
        y3 = self.up3(y2, e2) #decode3
        y4 = self.up4(y3, e1) #decode4
        out = self.out_conv(y4) #output
        if return_decoder_features:
            return out, y2, y3 #return_with_features

        return out #return_output

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="leaky_relu") #conv_init
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0) #bias_init
            elif isinstance(module, nn.InstanceNorm3d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1) #norm_weight_init
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0) #norm_bias_init


if __name__ == "__main__":
    model = UNet3D(n_channels=4, n_classes=3, base_filters=32)
    x = torch.randn(1, 4, 128, 128, 128)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
