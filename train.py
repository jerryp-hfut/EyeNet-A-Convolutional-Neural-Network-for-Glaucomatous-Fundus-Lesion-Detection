
'''
 *                             _ooOoo_
 *                            o8888888o
 *                            88" . "88
 *                            (| -_- |)
 *                            O\  =  /O
 *                         ____/`---'\____
 *                       .'  \\|     |//  `.
 *                      /  \\|||  :  |||//  \
 *                     /  _||||| -:- |||||-  \
 *                     |   | \\\  -  /// |   |
 *                     | \_|  ''\---/''  |   |
 *                     \  .-\__  `-`  ___/-. /
 *                   ___`. .'  /--.--\  `. . __
 *                ."" '<  `.___\_<|>_/___.'  >'"".
 *               | | :  `- \`.;`\ _ /`;.`/ - ` : | |
 *               \  \ `-.   \_ __\ /__ _/   .-` /  /
 *          ======`-.____`-.___\_____/___.-`____.-'======
 *                             `=---='
 *          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *                     佛祖保佑        永无BUG
'''


import torch
import os
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from models import model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.dataLoader import GlaucomaDataset

import predict

learning_rate = 0.01
# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为PyTorch张量
])

# 数据集路径
data_dir = 'datas/archive'

# 创建训练集和测试集的数据集实例
train_dataset = GlaucomaDataset(os.path.join(data_dir, 'train'), transform)

# 创建数据加载器，训练集打乱数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


md = model.EyeNet().cuda()
criterion = nn.BCELoss().cuda()
optimizer = optim.Adam(md.parameters(),lr = learning_rate)

def train(pre_epochs, epochs):
    losses = []
    accuracies = []
    max_accuracy = 0
    # 如果预训练过了就加载已有权重
    if pre_epochs >= 1:
        md.load_state_dict(torch.load('check_point.pth'))
    for epoch in range(pre_epochs, epochs):
        print("epoch=",epoch)
        for i,(features,label) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = md(features)
            outputs = outputs.squeeze()
            loss = criterion(outputs, label.float())  # 将标签转换为浮点型 CUDA出错位置0
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, epochs, i+1, len(train_loader), loss.item()))
            losses.append(loss.item())
        torch.save(md.state_dict(), 'check_point.pth') #每一个epoch训练结束保存一次模型
        acc = predict.eva()
        accuracies.append(acc)
        if acc > max_accuracy:
            max_accuracy = acc
            torch.save(md.state_dict(), f'best_model_acc_{str(acc)}.pth')
    plt.plot(losses)
    plt.xlabel("ITERATIONS")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
    
    plt.plot(accuracies)
    plt.xlabel("ITERATIONS")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.show()
    
if __name__ == '__main__':
    train(0, 200)