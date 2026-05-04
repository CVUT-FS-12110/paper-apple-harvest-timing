import torch
import torch.nn as nn

from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    densenet121, DenseNet121_Weights,
)


class RegressorModel(nn.Module):
    """
    Unified regression model for fair backbone comparison.

    Key property:
      - Encoder is swapped by name
      - A shared regression head is used
      - Feature dimension is inferred via a dummy forward, so no brittle in_features assumptions

    Supported backbones:
      - resnet18
      - resnet50
      - efficientnet_b2
      - convnext_tiny
      - mobilenet_v3_large
      - densenet121
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_outputs: int = 1,
        pretrained: bool = True,
        input_size: int = 512,   # used only to infer feature dim (match your training images)
        device: str | None = None,
    ):
        super().__init__()
        self.backbone = backbone.lower().strip()

        self.encoder = self._build_encoder(self.backbone, pretrained)

        # Infer encoder output feature dim robustly
        in_features = self._infer_in_features(input_size=input_size, device=device)

        # ---- Shared Regression Head ----
        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 1),
        )

    @staticmethod
    def _build_encoder(backbone: str, pretrained: bool) -> nn.Module:
        def w(weights_enum):
            return weights_enum if pretrained else None

        if backbone == "resnet18":
            m = resnet18(weights=w(ResNet18_Weights.IMAGENET1K_V1))
            m.fc = nn.Identity()
            return m

        if backbone == "resnet50":
            m = resnet50(weights=w(ResNet50_Weights.IMAGENET1K_V2))
            m.fc = nn.Identity()
            return m

        if backbone == "efficientnet_b2":
            m = efficientnet_b2(weights=w(EfficientNet_B2_Weights.IMAGENET1K_V1))
            m.classifier = nn.Identity()
            return m

        if backbone == "convnext_tiny":
            m = convnext_tiny(weights=w(ConvNeXt_Tiny_Weights.IMAGENET1K_V1))
            # keep LN + Flatten; remove final Linear only
            m.classifier[2] = nn.Identity()
            return m

        if backbone == "mobilenet_v3_large":
            m = mobilenet_v3_large(weights=w(MobileNet_V3_Large_Weights.IMAGENET1K_V1))
            # safest: remove classifier, keep everything else
            m.classifier = nn.Identity()
            return m

        if backbone == "densenet121":
            m = densenet121(weights=w(DenseNet121_Weights.IMAGENET1K_V1))
            m.classifier = nn.Identity()
            return m

        raise ValueError(
            f"Unknown backbone '{backbone}'. Supported: "
            "resnet18, resnet50, efficientnet_b2, convnext_tiny, mobilenet_v3_large, densenet121"
        )

    @torch.no_grad()
    def _infer_in_features(self, input_size: int, device: str | None) -> int:
        # Put encoder on same device temporarily for probing
        probe_device = device
        if probe_device is None:
            try:
                probe_device = next(self.encoder.parameters()).device.type
            except StopIteration:
                probe_device = "cpu"

        x = torch.zeros(1, 3, input_size, input_size, device=probe_device)

        self.encoder.eval()
        y = self.encoder(x)

        # handle possible shapes: [1,F] or [1,C,1,1] etc.
        if y.ndim == 4:
            y = y.flatten(1)
        elif y.ndim != 2:
            y = y.view(y.size(0), -1)

        return int(y.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if feats.ndim == 4:
            feats = feats.flatten(1)  # safety for any backbone returning [B,C,1,1]
        out = self.regressor(feats)
        return out.squeeze(-1)