import os
import sys
import torch
import torch.nn as nn
from .genconvit_ed import GenConViTED
from .genconvit_vae import GenConViTVAE
from torchvision import transforms
from torchvision.models.video import mc3_18

# Add the parent directory containing the 'model' module to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you should be able to import from the 'model' module
from model.cnn3d import CNN3D 

class GenConViT(nn.Module):

    def __init__(self, config, ed, vae, net, fp16, backbone, use_3dcnn=True):
        super(GenConViT, self).__init__()
        self.net = net
        self.fp16 = fp16
        self.use_3dcnn = use_3dcnn
        if self.net=='ed':
            try:
                self.model_ed = GenConViTED(config)
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=torch.device('cpu'))
                self.model_ed.load_state_dict(self.checkpoint_ed)
                self.model_ed.eval()
                if self.fp16:
                    self.model_ed.half()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{ed}.pth file not found.")
        elif self.net=='vae':
            try:
                self.model_vae = GenConViTVAE(config)
                self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=torch.device('cpu'))
                self.model_vae.load_state_dict(self.checkpoint_vae)
                self.model_vae.eval()
                if self.fp16:
                    self.model_vae.half()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{vae}.pth file not found.")
        else:
            try:
                self.model_ed = GenConViTED(config)
                self.model_vae = GenConViTVAE(config)
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=torch.device('cpu'))
                self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=torch.device('cpu'))
                self.model_ed.load_state_dict(self.checkpoint_ed)
                self.model_vae.load_state_dict(self.checkpoint_vae)
                self.model_ed.eval()
                self.model_vae.eval()
                if self.fp16:
                    self.model_ed.half()
                    self.model_vae.half()
            except FileNotFoundError as e:
                raise Exception(f"Error: Model weights file not found.")
            
        # Load backbone model (ConvNeXt or Swin Transformer)
        self.backbone = backbone

        # Initialize 3D CNN model if specified
        if self.use_3dcnn:
            self.cnn3d = CNN3D()

        # Define additional layers for integration
        self.fc = nn.Linear(self.num_features, self.num_features // 4)
        self.relu = nn.ReLU()

    def load_checkpoint(self, model, checkpoint):
        try:
            state_dict = torch.load(f'weight/{checkpoint}.pth', map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            if self.fp16:
                model.half()
        except FileNotFoundError:
            raise Exception(f"Error: weight/{checkpoint}.pth file not found.")


    def forward(self, x):
        if self.net == 'ed' :
            x = self.model_ed(x)
        elif self.net == 'vae':
            x,_ = self.model_vae(x)
        else:
            x1 = self.model_ed(x)
            x2,_ = self.model_vae(x)
           # x =  torch.cat((x1, x2), dim=0) #(x1+x2)/2 #
            
        # Pass through the backbone model (ConvNeXt or Swin Transformer)
        x_backbone = self.backbone(x)

        # Extract features from 3D CNN if specified
        if self.use_3dcnn:
            x_cnn3d = self.cnn3d(x)

        # Concatenate features from ED and VAE models
        if self.net != 'ed':
            x = torch.cat((x1, x2), dim=1)

        # Concatenate features from all sources
        combined_features = torch.cat((x, x_backbone), dim=1)

        # Add 3D CNN features if applicable
        if self.use_3dcnn:
            combined_features = torch.cat((combined_features, x_cnn3d), dim=1)

        # Pass through additional integration layers
        integrated_features = self.relu(self.fc(combined_features))

        return integrated_features
            