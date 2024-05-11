import torch
import torch.nn as nn

class EyeNet(nn.Module):
    def __init__(self):
        super(EyeNet,self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 ,stride=2),
            
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classify = nn.Sequential(
            nn.Linear(25*25*256,512),
            nn.ReLU(),
            torch.nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.Conv(x)
        x = x.view(-1, 25 * 25 * 256)
        x = self.classify(x)
        return x


