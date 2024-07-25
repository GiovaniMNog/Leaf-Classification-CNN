import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50WithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ResNet50WithDropout, self).__init__()
        self.resnet = resnet50(pretrained=True)
        
        # Insert dropout layers after each layer in the ResNet
        self.resnet.layer1 = nn.Sequential(
            self.resnet.layer1,
            nn.Dropout(dropout_rate)
        )
        self.resnet.layer2 = nn.Sequential(
            self.resnet.layer2,
            nn.Dropout(dropout_rate)
        )
        self.resnet.layer3 = nn.Sequential(
            self.resnet.layer3,
            nn.Dropout(dropout_rate)
        )
        self.resnet.layer4 = nn.Sequential(
            self.resnet.layer4,
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Create an instance of the model with a dropout 
model = ResNet50WithDropout(dropout_rate=0.5)
