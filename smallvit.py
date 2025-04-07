import torch.nn as nn
import timm
import torch


class ViTSmallCIFAR10(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super(ViTSmallCIFAR10, self).__init__()

        self.backbone = timm.create_model(
            'vit_small_patch16_224', pretrained=pretrained
        )

        # Replacing the classification head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = ViTSmallCIFAR10(pretrained=True, num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
