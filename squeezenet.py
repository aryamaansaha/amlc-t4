import torch.nn as nn
from torchvision.models import squeezenet1_1

def SqueezeNetCIFAR10(num_classes=10):
    model = squeezenet1_1(weights=None)
    
    # Remove the big stride-2 initial layer
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    
    # Replace classifier for CIFAR-10
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    
    return model
