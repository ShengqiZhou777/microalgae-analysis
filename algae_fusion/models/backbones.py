import torch
import torch.nn as nn
from torchvision import models

class ResNetRegressor(nn.Module):
    def __init__(self, backbone: str = "resnet18", in_channels: int = 3):
        super().__init__()
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if in_channels != 3:
            # Replace the first layer to support stacking multiple images
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 
                old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding, 
                bias=old_conv.bias
            )
            
            # Smart Initialization: Repeat pre-trained weights to maintain transfer learning benefits
            with torch.no_grad():
                # For 9 channels (3 images), we repeat the 3-channel weights 3 times
                # This ensures that each image in the stack is processed similarly initially
                n_repeats = in_channels // 3
                if in_channels % 3 == 0:
                    self.backbone.conv1.weight.copy_(old_conv.weight.repeat(1, n_repeats, 1, 1))
                else:
                    # Fallback for non-multiples: initialize with mean
                    avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    self.backbone.conv1.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))

        for name, param in self.backbone.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.6),  
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),   
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)
