import os
import re
import cv2
import numpy as np
import torch
import math
from random import randint
from scipy.spatial import Delaunay
from torch_geometric.utils import from_scipy_sparse_matrix
from TargetDetection import detection_target
from SiameseCNNnet import SiameseVGG, SiameseResNet18, SiameseResNet50, SiameseResNet101
from FunctionLib import *

# 对比方法的代码
# from TTS import SimilarityAggregationNeighbors_TTS
# from RTS import SimilarityAggregationNeighbors_RTS

# 获取每个节点最近的method个邻居，method=0使用Delaunay方法获取邻居节点
Neighbors = 4
# 迭代次数
Iters = 3
input_shape = [105, 105]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

if __name__ == '__main__':
    # 保存两个无人机拍摄图片的文件夹
    ImgAfold = './data/ImgA'
    ImgBfold = './data/ImgB'

    # 读取孪生神经网络权重提取视觉特征
    modelpath = './model/SiameseVGG11.pth'
    SiameseModel = Siamese(modelpath, input_shape, device)

    # 遍历图片文件夹中的所有文件
    ImgA_list = os.listdir(ImgAfold)
    for file in ImgA_list:
        print("处理{}".format(file))
        name = re.split('_|.j', file)

        # 获取图片文件路径
        ImgApath = os.path.join(ImgAfold, file)
        ImgBpath = os.path.join(ImgBfold, 'img_B_{}.jpg'.format(name[2]))

        # 读取图片
        ImgA = cv2.imread(ImgApath)
        ImgB = cv2.imread(ImgBpath)

        # 检测目标像素坐标TagInfX是目标X的识别信息
        # [0, 'people', tensor(1652.), tensor(1991.), tensor(1691.), tensor(2024.)]
        ImgA_, TagInfA = detection_target(ImgA, type=['car', 'van'])
        ImgB_, TagInfB = detection_target(ImgB, type=['car', 'van'])

        # 根据孪生神经网络获取视觉特征的相似度矩阵Mv
        Mv = GetSimilarityMatrix(ImgA, TagInfA, ImgB, TagInfB, SiameseModel)

        # 获取每个ID对应的像素坐标
        TagInfA = GetTargetCenter(TagInfA)
        TagInfB = GetTargetCenter(TagInfB)

        # 获取每个节点最近的method个邻居，method=0使用Delaunay方法获取邻居节点
        NeigA = GetNeighbors(TagInfA, method=Neighbors)
        NeigB = GetNeighbors(TagInfB, method=Neighbors)

        # 将2张图片拼接以支持后续显示关联结果
        Gimage = np.concatenate([ImgA_, ImgB_], axis=1)

        #########################
        # 下面开始获取关联结果
        #########################

        # 获取孪生神经网络的关联结果
        P = GetCorrelationResultsFromSimilarityMatrix(Mv)
        _ = ShowResult(Gimage, P, TagInfA, TagInfB, ImageName="Siamese")

        # 获取第一次聚合邻居的相似度
        Mu = SimilarityAggregationNeighbors(Mv, NeigA, NeigB, TagInfA, TagInfB)
        P = GetCorrelationResultsFromSimilarityMatrix(Mu)
        _ = ShowResult(Gimage, P, TagInfA, TagInfB, ImageName="Iters=1")

        # 重复迭代多次邻居聚合结果
        for i in range(2, 1+Iters):
            Mu = SimilarityAggregationNeighbors(Mu, NeigA, NeigB, TagInfA, TagInfB)
            P = GetCorrelationResultsFromSimilarityMatrix(Mu)
            Gimage_ = ShowResult(Gimage, P, TagInfA, TagInfB, ImageName="Iters={}".format(i))

        # 获取Va到Vb的变换矩阵
        H, P = GeTransformationMatrix(Mu, TagInfA, TagInfB)
        # 使用全局一致性检验，论文里的结果是没有使用这一步的
        P1 = GlobalConsistencyCheck(H, P, TagInfA, TagInfB, threshold=100)
        Gimage_ = ShowResult(Gimage, P1, TagInfA, TagInfB, ImageName="GlobalConsistency")

        cv2.waitKey()



