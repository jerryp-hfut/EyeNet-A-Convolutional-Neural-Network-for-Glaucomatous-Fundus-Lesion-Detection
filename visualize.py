import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from models import model
from utils import dataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import cv2
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE

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
    return predicted_prob.item(), diagnosis_result

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        target = output[:, 0]
        target.backward()

        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()

        weights = np.mean(gradients, axis=(2, 3))
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))  # Ensure cam matches the input image size
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

def visualize_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def plot_roc_curve(true_labels, predicted_probs):
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve-2022218058 Fang Yuhao')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
def plot_precision_recall_curve(true_labels, predicted_probs):
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
    plt.figure()
    plt.plot(recall, precision, marker='.', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve-2022218605 Wan Lizhi')
    plt.savefig('preRecall.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
def visualize_features_tsne(model, data_loader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            output = model.Conv(imgs)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            labels.append(lbls.cpu().numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    # t-SNE visualization
    perplexity = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_features = tsne.fit_transform(features)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=labels, palette='viridis')
    plt.title('t-SNE of Features-2022217940 Pan Xu')
    plt.show()
    plt.savefig('TSNE.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    
def plot_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Glaucoma'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix-2022218058 Fang Yuhao')
    plt.savefig('confusionMat.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def main():
    # 实例化模型并加载权重
    md = model.EyeNet().cuda()
    md.load_state_dict(torch.load("best_model.pth"))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize Grad-CAM with the target layer
    target_layer = md.Conv[6]  # 最后一个卷积层
    grad_cam = GradCAM(md, target_layer)
    
    val_dir = 'datas/archive/val'
    num_images = 200  # 设置使用的图片数量
    glaucoma_images = [os.path.join(val_dir, '1', img) for img in os.listdir(os.path.join(val_dir, '1'))[:num_images]]
    normal_images = [os.path.join(val_dir, '0', img) for img in os.listdir(os.path.join(val_dir, '0'))[:num_images]]
    
    # 实例化模型并加载权重


    
    results = []
    true_labels = []
    predicted_probs = []
    predicted_labels = []
    
    for img_path in glaucoma_images:
        prob, result = diagnosis(img_path, md, transform)
        results.append((img_path, result, 1))  # 1表示真实标签为青光眼阳性
        true_labels.append(1)
        predicted_probs.append(prob)
        predicted_labels.append(1 if prob >= 0.5 else 0)
    
    for img_path in normal_images:
        prob, result = diagnosis(img_path, md, transform)
        results.append((img_path, result, 0))  # 0表示真实标签为青光眼阴性
        true_labels.append(0)
        predicted_probs.append(prob)
        predicted_labels.append(1 if prob >= 0.5 else 0)
    
    plot_roc_curve(true_labels, predicted_probs)
    plot_confusion_matrix(true_labels, predicted_labels)
    plot_precision_recall_curve(true_labels, predicted_probs)

    img_path = "demo/resizedROI.png"
    img_plt = plt.imread(img_path)
    plt.imshow(img_plt)
    original_img = Image.open(img_path).convert('RGB')
    img_array = np.array(original_img)[:, :, ::-1]  # Convert to BGR for OpenCV
    img = dataLoader.getROI(original_img)
    img = transform(img).unsqueeze(0).cuda()
    cam = grad_cam(img)
    cam_img = visualize_cam_on_image(img_array, cam)
    plt.imshow(cam_img)
    plt.show()
    cv2.imwrite(f"gradcam_{os.path.basename(img_path)}.jpg", cam_img)

    val_dataset = []
    for img_path in glaucoma_images + normal_images:
        img = Image.open(img_path).convert('RGB')
        img = dataLoader.getROI(img)
        img = transform(img)
        label = 1 if '1' in img_path else 0
        val_dataset.append((img, label))
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False)
    visualize_features_tsne(md, val_loader)

# 运行主程序
if __name__ == "__main__":
    main()
