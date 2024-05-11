import torch
import os
from models import model
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.dataLoader import *

def eva():
    md = model.EyeNet().cuda()
    md.load_state_dict(torch.load("check_point.pth"))
    md.eval()

    transform = transforms.Compose([
        transforms.ToTensor()  # 转换为PyTorch张量
    ])
    data_dir = 'datas/archive'
    test_dataset = GlaucomaDataset(os.path.join(data_dir, 'test'), transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    close_count = 0
    total = 0
    tolerance = 0.5
    with torch.no_grad():
        for features, label in test_loader:
            output = md(features)
            output = output.squeeze()  # Ensure output is a 1D tensor for comparison
            total += label.size(0)
            difference = torch.abs(output - label.squeeze())  # Make sure label is also 1D for comparison
            close_count += (difference <= tolerance).sum().item()  # Correctly count predictions within tolerance
    if total > 0:
        close_ratio = close_count / total
        
        # print('Predictions within 0.2 of true value:', close_count)
        # print('Total samples:', total)
        print('Percentage of predictions within 0.2 of true value: {:.2f}%'.format(close_ratio*100.0))
        return close_ratio
    else:
        print('No samples to evaluate.')