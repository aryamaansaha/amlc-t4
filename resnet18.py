import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x))) #  (N, in_channels, H, W) -> (N, out_channels, H/stride, W/stride)
        out = self.bn2(self.conv2(out)) # (N, out_channels, H/stride, W/stride) -> (N, out_channels, H/stride, W/stride)
        out = out + self.skip_connection(x) # no shape change
        out = self.relu(out)
        return out

class SubGroup(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride):
        super(SubGroup, self).__init__()
        self.blocks = nn.Sequential(
            Block(in_channels, out_channels, stride),
            *[Block(out_channels, out_channels, stride=1) for _ in range(num_blocks-1)]
        )
    def forward(self, x):
        return self.blocks(x)
        

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.sg1 = SubGroup(in_channels=64, out_channels=64, num_blocks=2, stride=1)
        self.sg2 = SubGroup(in_channels=64, out_channels=128, num_blocks=2, stride=2)
        self.sg3 = SubGroup(in_channels=128, out_channels=256, num_blocks=2, stride=2)
        self.sg4 = SubGroup(in_channels=256, out_channels=512, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x))) # (N, 3, 32, 32) -> (N, 64, 32, 32)
        out = self.sg1(out) # (N, 64, 32, 32) -> (N, 64, 32, 32)
        out = self.sg2(out) # (N, 64, 32, 32) -> (N, 128, 16, 16)
        out = self.sg3(out) # (N, 128, 16, 16) -> (N, 256, 8, 8)
        out = self.sg4(out) # (N, 256, 8, 8) -> (N, 512, 4, 4)
        out = self.avgpool(out) # (N, 512, 4, 4) -> (N, 512, 1, 1)
        out = out.view(out.size(0), -1) # (N, 512, 1, 1) -> (N, 512)
        out = self.fc(out) # (N, 512) -> (N, 10)
        return out

