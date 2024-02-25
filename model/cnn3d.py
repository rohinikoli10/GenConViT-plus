import torch
import torch.nn as nn
from torchvision.models.video import mc3_18

class CNN3D(nn.Module):
    def __init__(self, num_classes=1000):
        super(CNN3D, self).__init__()
        # Load the MC3-18 model without pre-trained weights
        self.backbone = mc3_18(pretrained=False)

        # Modify the output dimension of the final classification layer to match the combined features
        self.backbone.fc = nn.Identity()

        # Additional layers for integration
        self.integration_layers = nn.Sequential(
            nn.Linear(512, 256),  # Adjust input and output dimensions as needed
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)   # Adjust input and output dimensions as needed
        )

    def forward(self, x):
        # Perform forward pass through the backbone (MC3-18)
        x = self.backbone(x)

        # Apply additional layers for integration
        x = self.integration_layers(x)

        return x
