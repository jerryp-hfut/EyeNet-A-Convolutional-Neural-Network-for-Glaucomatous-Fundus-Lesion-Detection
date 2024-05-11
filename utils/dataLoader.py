import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

def getROI(image):
    if isinstance(image, Image.Image):  # 检查image是否为PIL图像
        image = np.array(image)  # 将PIL图像转换为numpy数组
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (65, 65), 0)

    # Find the pixel with the highest intensity value
    max_intensity_pixel = np.unravel_index(np.argmax(blurred_image), gray_image.shape)

    # Define the radius for the circle
    radius = 200 // 2

    # Get the x and y coordinates for cropping the image
    x = max_intensity_pixel[1] - radius
    y = max_intensity_pixel[0] - radius

    # Ensure the coordinates are within the image boundaries
    x = max(0, x)
    y = max(0, y)

    # Define the size of the outer square
    square_size = 2 * radius

    # Create a mask for the outer square
    mask = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + square_size, y + square_size), (255, 255, 255), -1)

    # Apply the mask to the original image to get the ROI
    roi_image = cv2.bitwise_and(image, mask)

    # Crop the outer square
    cropped_roi = roi_image[y:y+square_size, x:x+square_size]

    # Resize the cropped ROI to a fixed size square if needed
    resized_roi = cv2.resize(cropped_roi, (square_size, square_size))
    
    return resized_roi

class GlaucomaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 获取所有图像的路径和标签
        self.images = []
        self.labels = []
        for label in [0, 1]:
            label_dir = os.path.join(self.data_dir, str(label))
            for img_file in os.listdir(label_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(label_dir, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.images[idx]).convert('RGB')  # 确保图像是RGB格式
        # 应用自定义的getROI函数
        image = getROI(image)
        
        # 应用转换操作
        if self.transform:
            image = self.transform(image)
            image = image.cuda()
        
        # 标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        label.cuda()
        return image, label