"""Loss functions for binary segmentation models with deep supervision."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Binary cross entropy combined with soft Dice loss.

    ``U3Net`` returns sigmoid probabilities, so this loss intentionally uses
    binary cross entropy instead of ``BCEWithLogitsLoss``.
    """

    def __init__(
            self,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
            smooth: float = 1.0,
            eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight == 0:
            raise ValueError("loss weights must be non-negative and not both zero")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
            )

        # PyTorch intentionally rejects BCELoss while autocast is active.  This
        # model returns sigmoid probabilities (not logits), so keep BCE and the
        # numerically sensitive Dice reductions in an explicit FP32 region.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float().clamp(self.eps, 1.0 - self.eps)
            target = target.float()

            bce = nn.functional.binary_cross_entropy(prediction, target)
            dims = tuple(range(1, prediction.ndim))
            intersection = (prediction * target).sum(dim=dims)
            denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
            dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            dice_loss = 1.0 - dice.mean()

            normalizer = self.bce_weight + self.dice_weight
            return (self.bce_weight * bce + self.dice_weight * dice_loss) / normalizer


class DeepSupervisionLoss(nn.Module):
    """Apply segmentation loss to the fused and six side predictions.

    By default the fused prediction receives half of the total weight, while
    the remaining weight is distributed evenly over all side predictions.
    """

    def __init__(
            self,
            fused_weight: float = 1.0,
            side_weight: float = 1.0 / 6.0,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if fused_weight <= 0 or side_weight < 0:
            raise ValueError("fused_weight must be positive and side_weight non-negative")
        self.fused_weight = fused_weight
        self.side_weight = side_weight
        self.base_loss = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

    def forward(
            self,
            predictions: torch.Tensor | Sequence[torch.Tensor],
            target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(predictions, torch.Tensor):
            predictions = (predictions,)
        if len(predictions) == 0:
            raise ValueError("predictions must contain at least one tensor")

        weights = [self.fused_weight] + [self.side_weight] * (len(predictions) - 1)
        losses = [
            weight * self.base_loss(prediction, target)
            for weight, prediction in zip(weights, predictions)
        ]
        return torch.stack(losses).sum() / sum(weights)
