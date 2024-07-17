import random
import torch
import os
import pickle
import cv2
import re
import torch.nn as nn
import numpy as np
from random import randint

# from tools.TargetDetection import detection_target
from SiameseCNNtrain import SiameseDataset
from torch.utils.data import Dataset, DataLoader

from SiameseCNNnet import SiameseVGG, SiameseResNet18, SiameseResNet50, SiameseResNet101

# 输入图像的大小，默认为210,210,3
input_shape = [105, 105]
image_folder = 'H:\GNNProject\SiameseGNN\data\MDMT\\val'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

def ReadTxt(path):
    txtA = open(path, 'r')
    dataA = []
    for line in txtA:
        data = line.split('\t')
        dataA.append([data[0], data[1], data[2], data[3], data[4], int(data[5]), data[6]])
    txtA.close()
    return dataA

class Siamese(object):
    def __init__(self, modelpath, input_shape) -> None:
        super(Siamese, self).__init__()
        self.modelpath = modelpath
        self.device = device
        self.input_shape = input_shape
        self.model = SiameseVGG(input_shape).to(device)
        self.model.load_state_dict(torch.load(modelpath))

    def detect_image(self, imageA, imageB):
        pre = []
        imageA = torch.tensor([imageA]).to(self.device)
        imageB = torch.tensor([imageB]).to(self.device)

        imageA = imageA.permute(0, 3, 1, 2)
        imageB = imageB.permute(0, 3, 1, 2)
        # print(imageA.shape)

        output, _, _ = self.model([imageA.to(torch.float32), imageB.to(torch.float32)])
        # print(output)
        pre = torch.nn.Sigmoid()(output[0])

        return pre

def GetSimilarityMatrix(imageA, TxtA, imageB, TxtB, model):
    # print(TxtA)
    P = []
    imA = []
    imB = []
    SimM = np.zeros((len(TxtA), len(TxtB)))
    for i in range(len(TxtA)):
        A = TxtA[i]
        for j in range(len(TxtB)):
            B = TxtB[j]

            imgAsub = imageA[int(A[3]):int(A[5]), int(A[2]):int(A[4])]
            imgBsub = imageB[int(B[3]):int(B[5]), int(B[2]):int(B[4])]

            imgAsub = cv2.resize(imgAsub, input_shape)
            imgBsub = cv2.resize(imgBsub, input_shape)

            SimM[i][j] = model.detect_image(imgAsub, imgBsub)

        # if A[6] != -1:
        #     idxB = int(A[6])
        #     SimM[i][idxB] = SimM[i][idxB] + 1

    return SimM

def ShowCorrelationResults(Gimage, SimM, TxtA, TxtB):
    Res = []
    # print(SimM)
    # 获取关联配对的索引
    L = min(SimM.shape[0], SimM.shape[1])
    for i in range(L):
        sm = SimM.max()
        if sm < 0.8:
            break
        else:
            idx = np.unravel_index(SimM.argmax(), SimM.shape)
            Res.append(idx)
            SimM[idx[0], :] = 0
            SimM[:, idx[1]] = 0
    # print(Res)
    num = 0
    thickness = 2
    lineType = 100
    for D in Res:
        [idxA, idxB] = D
        # 每一对目标都生成不同的颜色
        Gcolor = (randint(0, 255), randint(0, 255), randint(0, 255))
        # 左边图片的点
        ptStart = (int(0.5 * (int(TxtA[idxA][2]) + int(TxtA[idxA][4]))),
               int(0.5 * (int(TxtA[idxA][3]) + int(TxtA[idxA][5]))))
        # 右边图片的点
        ptEnd = (3840 + int(0.5 * (int(TxtB[idxB][2]) + int(TxtB[idxB][4]))),
                 int(0.5 * (int(TxtB[idxB][3]) + int(TxtB[idxB][5]))))
        # 画点
        cv2.circle(Gimage, ptStart, 50, Gcolor, thickness=-1)
        cv2.circle(Gimage, ptEnd, 50, Gcolor, thickness=-1)
        # 画线
        cv2.line(Gimage, ptStart, ptEnd, Gcolor, thickness, lineType)

        # print(TxtA[idxA][6])
        if int(TxtA[idxA][6]) == idxB:
            num += 1

    print("关联正确率：{:.2f}%".format(100 * num/len(Res)))

    return Gimage, 100 * num/len(Res)

if __name__ == "__main__":
    modelpath = './model/SiameseVGG11epoch_1_i_18000_loss_0.194.pth'
    model = Siamese(modelpath, input_shape)

    ImgApath = 'H:\GNNProject\SiameseGNN\data\SiameseVGGdata\ImgA'
    ImgBpath = 'H:\GNNProject\SiameseGNN\data\SiameseVGGdata\ImgB'
    ImgAtxtpath = 'H:\GNNProject\SiameseGNN\data\SiameseVGGdata\TxtA_'
    ImgBtxtpath = 'H:\GNNProject\SiameseGNN\data\SiameseVGGdata\TxtB_'

    f = open("./result/mdmtvgg11res.txt", 'w')
    file_list = os.listdir(ImgApath)
    for file in file_list:
        print("\n处理图片：{}".format(file))
        name = re.split('_|.j', file)

        imgBname = '{}_B_{}.jpg'.format(name[0], name[2])
        txtAname = '{}_{}_{}.txt'.format(name[0], name[1], name[2])
        txtBname = '{}_B_{}.txt'.format(name[0], name[2])

        imgA_file_path = os.path.join(ImgApath, file)
        txtA_file_path = os.path.join(ImgAtxtpath, txtAname)

        imgB_file_path = os.path.join(ImgBpath, imgBname)
        txtB_file_path = os.path.join(ImgBtxtpath, txtBname)

        # print(imgA_file_path, txtA_file_path, imgB_file_path, txtB_file_path)

        imageA = cv2.imread(imgA_file_path)
        imageB = cv2.imread(imgB_file_path)

        TxtA = ReadTxt(txtA_file_path)
        TxtB = ReadTxt(txtB_file_path)

        SimM = GetSimilarityMatrix(imageA, TxtA, imageB, TxtB, model)

        # 核心拼接代码
        Gimage = np.concatenate([imageA, imageB], axis=1)

        Gimage, P = ShowCorrelationResults(Gimage, SimM, TxtA, TxtB)
        f.write("{}\t{}\n".format(file, P))
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1920, 540)
        cv2.imshow('image', Gimage)

        cv2.waitKey()
        # num += 1
        # if num > 5:
        #     break
    f.close()
