"""
SpeechBrain ECAPA-TDNN Architecture for Speaker Verification.
Matches official weights from speechbrain/spkrec-ecapa-voxceleb.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dBlock(nn.Module):
    """
    SpeechBrain-compatible 1D Convolution block with BatchNorm and ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, dilation: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Module()
        self.conv.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.norm = nn.Module()
        self.norm.norm = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm.norm(self.conv.conv(x)))


class Res2NetBlock(nn.Module):
    """
    Res2Net hierarchical multi-scale convolution block.
    """
    def __init__(self, in_channels: int, out_channels: int, scale: int = 8, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.scale = scale
        self.width = in_channels // scale
        self.blocks = nn.ModuleList([
            Conv1dBlock(
                self.width,
                self.width,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) // 2 * dilation
            )
            for _ in range(scale - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.width, dim=1)
        out = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                out.append(chunk)
            elif i == 1:
                out.append(self.blocks[i - 1](chunk))
            else:
                out.append(self.blocks[i - 1](chunk + out[i - 1]))
        return torch.cat(out, dim=1)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation channel attention block.
    """
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.conv1 = nn.Module()
        self.conv1.conv = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Module()
        self.conv2.conv = nn.Conv1d(bottleneck, channels, kernel_size=1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2, keepdim=True)
        s = self.act(self.conv1.conv(s))
        s = self.act2(self.conv2.conv(s))
        return x * s


class SERes2NetBlock(nn.Module):
    """
    Squeeze-and-Excitation Res2Net Block with residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, scale: int = 8):
        super().__init__()
        self.tdnn1 = Conv1dBlock(in_channels, out_channels, kernel_size=1, dilation=1, padding=0)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, scale=scale, kernel_size=kernel_size, dilation=dilation)
        self.tdnn2 = Conv1dBlock(out_channels, out_channels, kernel_size=1, dilation=1, padding=0)
        self.se_block = SqueezeExcitation(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)
        return x + res


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP) layer with global context.
    """
    def __init__(self, channels: int = 3072, attention_channels: int = 128, global_context_att: bool = True):
        super().__init__()
        self.global_context_att = global_context_att
        in_dim = channels * 3 if global_context_att else channels
        self.tdnn = Conv1dBlock(in_dim, attention_channels, kernel_size=1, dilation=1, padding=0)
        self.conv = nn.Module()
        self.conv.conv = nn.Conv1d(attention_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Time)
        if self.global_context_att:
            mean = x.mean(dim=2, keepdim=True)
            std = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-12)
            T = x.shape[2]
            global_context = torch.cat([x, mean.expand(-1, -1, T), std.expand(-1, -1, T)], dim=1)
        else:
            global_context = x

        attn = self.tdnn(global_context)
        attn = self.conv.conv(attn)
        attn = F.softmax(attn, dim=2)
        mean = torch.sum(attn * x, dim=2)
        var = torch.sum(attn * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    """
    SpeechBrain ECAPA-TDNN Model for extracting 192-dimensional speaker embeddings.
    """
    def __init__(self, in_channels: int = 80, channels: int = 1024, lin_neurons: int = 192):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, channels, kernel_size=5, dilation=1, padding=2),
            SERes2NetBlock(channels, channels, kernel_size=3, dilation=2),
            SERes2NetBlock(channels, channels, kernel_size=3, dilation=3),
            SERes2NetBlock(channels, channels, kernel_size=3, dilation=4),
        ])
        self.mfa = Conv1dBlock(channels * 3, channels * 3, kernel_size=1, dilation=1, padding=0)
        self.asp = AttentiveStatisticsPooling(channels * 3, attention_channels=128)
        self.asp_bn = nn.Module()
        self.asp_bn.norm = nn.BatchNorm1d(channels * 6, eps=1e-5, momentum=0.1)
        self.fc = nn.Module()
        self.fc.conv = nn.Conv1d(channels * 6, lin_neurons, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, 80, Time)
        Returns:
            Tensor of shape (Batch, 192)
        """
        out1 = self.blocks[0](x)
        out2 = self.blocks[1](out1)
        out3 = self.blocks[2](out2)
        out4 = self.blocks[3](out3)
        mfa_in = torch.cat([out2, out3, out4], dim=1)
        mfa_out = self.mfa(mfa_in)
        pooled = self.asp(mfa_out)
        pooled_bn = self.asp_bn.norm(pooled)
        emb = self.fc.conv(pooled_bn.unsqueeze(-1)).squeeze(-1)
        return emb
