import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from models import model
from utils import dataLoader
import matplotlib.pyplot as plt
import os

def diagnosis(path, md, transform):
    md.eval()  # 模型处于评估模式（不训练）

    img = Image.open(path).convert('RGB')  # 确保图片是RGB格式
    img = dataLoader.getROI(img)
    img = transform(img)  # 应用转换操作
    img = torch.unsqueeze(img, 0)
    img = img.cuda()

    with torch.no_grad():
        output = md(img)
        output = output.squeeze()
        predicted_prob = torch.sigmoid(output)
        # 根据概率确定诊断结果
        diagnosis_result = 'Normal' if predicted_prob.item() < 0.5 else 'Glaucoma'

    return diagnosis_result

def visualize_results(results):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, (img_path, result, true_label) in enumerate(results):
        img = Image.open(img_path).convert('RGB')
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"model: {result}\nreal: {'Glaucoma' if true_label == 1 else 'Normal'}")
    plt.tight_layout()
    plt.show()

def main():
    # 实例化模型并加载权重
    md = model.EyeNet().cuda()
    md.load_state_dict(torch.load("best_model.pth"))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dir = 'datas/archive/val'
    glaucoma_images = [os.path.join(val_dir, '1', img) for img in os.listdir(os.path.join(val_dir, '1'))[:4]]
    normal_images = [os.path.join(val_dir, '0', img) for img in os.listdir(os.path.join(val_dir, '0'))[:4]]
    
    results = []
    
    for img_path in glaucoma_images:
        result = diagnosis(img_path, md, transform)
        results.append((img_path, result, 1))  # 1表示真实标签为青光眼阳性
    
    for img_path in normal_images:
        result = diagnosis(img_path, md, transform)
        results.append((img_path, result, 0))  # 0表示真实标签为青光眼阴性
    
    visualize_results(results)

# 运行主程序
if __name__ == "__main__":
    main()
