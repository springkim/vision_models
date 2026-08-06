"""Modern, dependency-free encoder/decoder model for dense prediction.

The public ``U2NET`` name and its seven sigmoid outputs are kept for compatibility
with training code written for the original U2-Net implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _upsample_like(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize ``src`` to the spatial resolution of ``target``."""
    return F.interpolate(src, size=target.shape[-2:], mode="bilinear", align_corners=False)


class DropPath(nn.Module):
    """Per-sample stochastic depth (no torchvision dependency)."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class GlobalResponseNorm(nn.Module):
    """ConvNeXt V2 global response normalization for NCHW tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(x.float(), ord=2, dim=(-2, -1), keepdim=True)
        response = response.to(dtype=x.dtype)
        normalized = response / (response.mean(dim=1, keepdim=True) + 1e-6)
        return x + self.gamma * (x * normalized) + self.beta


class ConvNeXtV2Block(nn.Module):
    """Depthwise ConvNeXt V2 block with GRN, layer scale and stochastic depth."""

    def __init__(self, channels: int, drop_path: float = 0.0, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        # GroupNorm(1, C) is LayerNorm-like but preserves the efficient NCHW layout.
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.activation = nn.GELU()
        self.grn = GlobalResponseNorm(hidden)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.layer_scale = nn.Parameter(1e-6 * torch.ones(1, channels, 1, 1))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.project(x)
        return residual + self.drop_path(self.layer_scale * x)


class ChannelGate(nn.Module):
    """Lightweight squeeze/excitation used after skip-feature fusion."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class Stage(nn.Module):
    def __init__(self, channels: int, depth: int, drop_paths: list[float]) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(ConvNeXtV2Block(channels, drop_paths[i]) for i in range(depth))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            depth: int,
            drop_paths: list[float],
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(in_channels, out_channels, 1)
        self.skip_projection = nn.Conv2d(skip_channels, out_channels, 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            ChannelGate(out_channels),
        )
        self.stage = Stage(out_channels, depth, drop_paths)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = _upsample_like(self.input_projection(x), skip)
        skip = self.skip_projection(skip)
        return self.stage(self.fusion(torch.cat((x, skip), dim=1)))


class U3Net(nn.Module):
    """ConvNeXt-V2-style U-Net with deep supervision.

    Args:
        in_ch: Number of input image channels.
        out_ch: Number of output channels/classes.
        widths: Channel width at each of the six feature scales.
        depths: Block count at each encoder scale.
        drop_path_rate: Maximum stochastic-depth probability.

    Returns:
        A tuple ``(fused, side1, ..., side6)``. Every tensor is sigmoid-normalized
        and has the same spatial shape as the input, matching legacy U2-Net code.
    """

    def __init__(
            self,
            in_ch: int = 3,
            out_ch: int = 1,
            widths: tuple[int, ...] = (48, 96, 192, 384, 512, 512),
            depths: tuple[int, ...] = (2, 2, 3, 3, 3, 3),
            drop_path_rate: float = 0.15,
    ) -> None:
        super().__init__()
        if len(widths) != 6 or len(depths) != 6:
            raise ValueError("widths and depths must each contain exactly six values")
        if min(widths) <= 0 or min(depths) <= 0:
            raise ValueError("all widths and depths must be positive")

        total_blocks = sum(depths) + sum(depths[:-1])
        rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cursor = 0

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, widths[0], 3, padding=1, bias=False),
            nn.GroupNorm(1, widths[0]),
        )
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (width, depth) in enumerate(zip(widths, depths)):
            self.encoder.append(Stage(width, depth, rates[cursor: cursor + depth]))
            cursor += depth
            if index < len(widths) - 1:
                self.downsamples.append(Downsample(width, widths[index + 1]))

        self.decoder = nn.ModuleList()
        current_width = widths[-1]
        for index in range(4, -1, -1):
            depth = depths[index]
            self.decoder.append(
                DecoderStage(
                    current_width,
                    widths[index],
                    widths[index],
                    depth,
                    rates[cursor: cursor + depth],
                )
            )
            cursor += depth
            current_width = widths[index]

        self.side_heads = nn.ModuleList(nn.Conv2d(width, out_ch, 1) for width in widths)
        # Softmax weights provide stable, interpretable learned side-output fusion.
        self.fusion_logits = nn.Parameter(torch.zeros(len(widths)))
        self.refine = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.ndim != 4:
            raise ValueError(f"expected a 4D NCHW tensor, got shape {tuple(x.shape)}")
        if min(x.shape[-2:]) < 32:
            raise ValueError("input height and width must both be at least 32")

        input_size = x.shape[-2:]
        features: list[torch.Tensor] = []
        x = self.stem(x)
        for index, stage in enumerate(self.encoder):
            x = stage(x)
            features.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        decoded = features[-1]
        decoded_features = [decoded]
        for decoder, skip in zip(self.decoder, reversed(features[:-1])):
            decoded = decoder(decoded, skip)
            decoded_features.append(decoded)

        # Reorder from finest to coarsest, as expected by the original U2NET API.
        pyramid = list(reversed(decoded_features))
        side_logits = [
            F.interpolate(head(feature), size=input_size, mode="bilinear", align_corners=False)
            for head, feature in zip(self.side_heads, pyramid)
        ]
        weights = self.fusion_logits.softmax(dim=0)
        fused_logits = self.refine(
            torch.stack([weight * side for weight, side in zip(weights, side_logits)]).sum(0)
        )
        return tuple(torch.sigmoid(logit) for logit in (fused_logits, *side_logits))


def modern_u2net_tiny(in_ch: int = 3, out_ch: int = 1) -> U3Net:
    """Smaller configuration for limited memory or fast experimentation."""
    return U3Net(
        in_ch=in_ch,
        out_ch=out_ch,
        widths=(32, 64, 128, 256, 320, 320),
        depths=(1, 1, 2, 2, 2, 2),
        drop_path_rate=0.1,
    )


if __name__ == "__main__":
    model = U3Net(in_ch=3, out_ch=1)
    model.eval()

    dummy_input = torch.randn(1, 3, 600, 600)

    torch.onnx.export(
        model,
        dummy_input,
        "modern_u2net.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input": {
                0: "batch_size",
                2: "height",
                3: "width",
            },
            "output": {
                0: "batch_size",
                2: "height",
                3: "width",
            },
        },
    )

    print("ONNX 모델 저장 완료: modern_u2net.onnx")
