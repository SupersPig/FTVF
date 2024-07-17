'''
参考：https://deepinout.com/pytorch/pytorch-questions/149_pytorch_siamese_neural_network_in_pytorch.html

'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import re
import pickle

from tools.DataProcess import SiameseDataset
from torch.utils.data import Dataset, DataLoader
from SiameseCNNnet import SiameseVGG, SiameseResNet18, SiameseResNet50, SiameseResNet101

# 输入图像的大小，默认为210,210,3
input_shape = [105, 105]
# image_folder = './data/SiameseVGGdata'
image_folder = 'H:\GNNProject\SiameseGNN\data\MDMT\\train'
dataset_filename = os.path.join(image_folder, 'mdmtrain.pkl')

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    Epoch = 1
    batch_size = 64
    learning_rate = 1e-3
    minloss = 0.03
    transform = None

    if os.path.exists(dataset_filename):
        print("使用已经存在的dataset文件")
        # 以二进制读模式打开目标文件
        f = open(dataset_filename, 'rb')
        # 将文件中的变量加载到当前工作区
        dataset = pickle.load(f)
        f.close()
    else:
        print("处理数据集...")
        f = open(dataset_filename, 'wb')
        # 创建数据集
        dataset = SiameseDataset(image_folder, transform=transform)
        pickle.dump(dataset, f)
        # 关闭文件
        f.close()

    print(dataset.__len__(), len(dataset.label), len(dataset.label_))
    # 将数据集分割成训练集和验证集
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # dataset = dataset.to(device)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    # 创建孪生神经网络
    # model = SiameseVGG(input_shape).to(device)
    model = SiameseResNet101(input_shape).to(device)
    #
    # 使用交叉熵损失函数和随机梯度下降（SGD）优化器
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    txtfile = "./result/res101trainlog.txt"
    f = open(txtfile, 'w')
    f.write('epoch\tdata_i\tloss\n')
    # 训练网络
    for epoch in range(Epoch):
        running_loss = 0.0
        L = 0.0
        PreL = 0.0
        for i, data in enumerate(dataloader):
            inputs1, inputs2, labels = data
            # print(i, labels[0])
            inputs1 = inputs1.permute(0, 3, 1, 2)
            inputs2 = inputs2.permute(0, 3, 1, 2)

            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            # print(inputs1.shape)
            optimizer.zero_grad()

            # 前向传播
            outputs, _, _ = model([inputs1.to(torch.float32), inputs2.to(torch.float32)])

            # print(outputs)
            # print(labels)
            labels = torch.tensor(labels, dtype=float)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            L = loss.item()
            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                f.write('{}\t{}\t{:.5f}\n'.format(epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        # if minloss > L or PreL > L:
            if i % 3000 == 0 and i > 2:
                PATH = './result/model/SiameseRes101epoch_{}_i_{}_loss_{:.3f}.pth'.format(epoch + 1, i, L)
                torch.save(model.state_dict(), PATH)
            if i > 18005:
                break
        PATH = './result/model/SiameseRes101epoch_{}_loss_{:.3f}.pth'.format(epoch + 1, L)
        torch.save(model.state_dict(), PATH)
        break
    f.close()


