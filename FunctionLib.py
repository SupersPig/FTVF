import math
import numpy as np
import cv2
import torch
import math
from random import randint
from scipy.spatial import Delaunay
from SiameseCNNnet import SiameseVGG
from SiameseCNNnet import SiameseResNet101 as SiameseResNet
# , SiameseResNet50, SiameseResNet101

def detection_target_fromtxt(TxtApath, TxtBpath, device):
    fa = open(TxtApath, 'r')
    fb = open(TxtBpath, 'r')
    TagInfA = []
    TagInfB = []
    P = []
    for line in fa:
        data = line.split('\t')
        TagInfA.append([int(data[0]), data[1], torch.tensor(float(data[2]), device=device),
                        torch.tensor(float(data[3]), device=device),
                        torch.tensor(float(data[4]), device=device),
                        torch.tensor(float(data[5]), device=device)])
        if int(data[6]) != -1:
            P.append([int(data[0]), int(data[6])])

    for line in fb:
        data = line.split('\t')
        TagInfB.append([int(data[0]), data[1], torch.tensor(float(data[2]), device=device),
                       torch.tensor(float(data[3]), device=device),
                       torch.tensor(float(data[4]), device=device),
                       torch.tensor(float(data[5]), device=device)])

    return P, TagInfA, TagInfB

def CalculateThCoordinate(Coordinate, K):
    res = np.dot(K, Coordinate)
    return res/res[2]

# 计算方位角函数
def AzimuthAngle(x1, y1, x2, y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)

def AzimuthalRotation(Azim, thae):
    return [AzimuthsPlusMinus(D, thae) for D in Azim]

# 返回D减去thae之后的方位角
def AzimuthsPlusMinus(D, thae):
    if D - thae < 0:
        return 360 + D - thae
    else:
        return D - thae

def GetAzimuthCorrelationsAndDeviations(NeigAazim_, NeigBazim_):
    P = []
    num = 0
    for Da in NeigAazim_:
        dis = []
        for Db in NeigBazim_:
            dis_ = max([Da, Db]) - min([Da, Db])
            if dis_ < 180:
                dis.append(dis_)
            else:
                dis.append(360 - dis_)
        idxmin = np.argmin(dis)
        valmin = np.min(dis)

        P.append([num, idxmin, valmin])
        num += 1
    return P

def GetAzimuthWeights(AzimDev):
    if AzimDev < 5:
        w = 1.0
    elif AzimDev > 15:
        w = 0
    else:
        w = (15 - AzimDev) * 0.1
    return w

def CalculatingSimilarity(P, Mv, NeigA_, NeigB_):
    m = 0
    for D in P:
        idxA, idxB, AzimDev = D
        m_ = Mv[NeigA_[idxA]][NeigB_[idxB]]
        w = GetAzimuthWeights(AzimDev)
        m += m_ if m_ < 0 else m_ * w
    return m / len(P)

def GlobalConsistencyCheck(H, P, TagInfA, TagInfB, threshold=100):
    P_ = []
    La = len(TagInfA)
    Lb = len(TagInfB)
    Fa = np.zeros((La))
    Fb = np.zeros((Lb))
    for D in P:
        CoA = [TagInfA[D[0]][1], TagInfA[D[0]][2]]
        CoB = [TagInfB[D[1]][1], TagInfB[D[1]][2]]
        CoA_ = CalculateThCoordinate([[CoA[0]], [CoA[1]], [1.0]], H)
        dis = math.sqrt(math.pow(CoA_[0] - CoB[0], 2) +
                        math.pow(CoA_[1] - CoB[1], 2))
        # print(dis, D)
        if dis < threshold:
            Fa[D[0]] = 1
            Fb[D[1]] = 1
            P_.append(D)
    # print(Fa, Fb, P_)
    for idxA in range(La):
        if Fa[idxA] == 0:
            for idxB in range(Lb):
                if Fb[idxB] == 0:
                    # CoA = TagInfA[idxA]
                    # CoB = TagInfB[idxB]
                    CoA = [TagInfA[idxA][1], TagInfA[idxA][2]]
                    CoB = [TagInfB[idxB][1], TagInfB[idxB][2]]
                    CoA_ = CalculateThCoordinate([[CoA[0]], [CoA[1]], [1.0]], H)
                    dis = math.sqrt(math.pow(CoA_[0] - CoB[0], 2) +
                                    math.pow(CoA_[1] - CoB[1], 2))

                    if dis < threshold:
                        # Fa[idxA] = 1
                        # Fb[idxB] = 1
                        P_.append((idxA, idxB))
                        # print(dis, (idxA, idxB))

    return P_

input_shape = [105, 105]

class Siamese(object):
    def __init__(self, modelpath, input_shape, device) -> None:
        super(Siamese, self).__init__()
        self.modelpath = modelpath
        self.device = device
        self.input_shape = input_shape
        self.model = SiameseVGG(input_shape).to(device)
        self.model.load_state_dict(torch.load(modelpath))

    def detect_image(self, imageA, imageB):
        pre = []
        imageA = torch.tensor(imageA).to(self.device)
        imageB = torch.tensor(imageB).to(self.device)
        # print(imageA.shape)
        # print(imageB.shape)
        imageA = imageA.permute(0, 3, 1, 2)
        imageB = imageB.permute(0, 3, 1, 2)
        output, x1, x2 = self.model([imageA.to(torch.float32), imageB.to(torch.float32)])
        Sim = np.zeros((len(x1), len(x2)))
        for i in range(len(x1)):
            for j in range(len(x2)):
                x = torch.abs(x1[i] - x2[j])
                output = self.model.fully_connect2(x)
                # 是否转化为百分制相似度
                Sim[i][j] = torch.nn.Sigmoid()(output[0])
        # pre = output[0]
        return Sim

# 根据目标检测框获取目标的视觉相似度矩阵
# def GetSimilarityMatrix(imageA, TxtA, imageB, TxtB, model):
#     SimM = np.zeros((len(TxtA), len(TxtB)))
#     for i in range(len(TxtA)):
#         A = TxtA[i]
#         imgAsub = imageA[int(A[3]):int(A[5]), int(A[2]):int(A[4])]
#         imgAsub = cv2.resize(imgAsub, input_shape)
#         for j in range(len(TxtB)):
#             B = TxtB[j]
#             imgBsub = imageB[int(B[3]):int(B[5]), int(B[2]):int(B[4])]
#             imgBsub = cv2.resize(imgBsub, input_shape)
#             SimM[i][j] = model.detect_image(imgAsub, imgBsub)
#     return SimM

# 根据目标检测框获取目标的视觉相似度矩阵
def GetSimilarityMatrix(imageA, TxtA, imageB, TxtB, model):
    imgA = []
    imgB = []
    for i in range(len(TxtA)):
        A = TxtA[i]
        imgAsub = imageA[int(A[3]):int(A[5]), int(A[2]):int(A[4])]
        imgAsub = cv2.resize(imgAsub, input_shape)
        imgA.append(imgAsub)
    for j in range(len(TxtB)):
        B = TxtB[j]
        imgBsub = imageB[int(B[3]):int(B[5]), int(B[2]):int(B[4])]
        imgBsub = cv2.resize(imgBsub, input_shape)
        imgB.append(imgBsub)
    SimM = model.detect_image(imgA, imgB)
    return SimM

def GetTargetCenter(TagInf):
    Res = []
    for D in TagInf:
        Res.append([D[0], 0.5 * float(D[2] + D[4]), 0.5 * float(D[3] + D[5])])
    return Res

# method=0表示使用Delaunay方法获取邻居节点，method>0表示最近的method个点做邻居
def GetNeighbors(TagInf, method=0):
    Res = []
    Nodenum = len(TagInf)
    nodepos = []
    for D in TagInf:
        nodepos.append([D[1], D[2]])
    if method > 0:
        # 使用最近邻N的方式获取邻居节点，N=method
        lj = method
        if Nodenum < lj:
            A = np.ones((Nodenum, Nodenum)) - np.eye(Nodenum)
        else:
            A = np.zeros((Nodenum, Nodenum))
            for idx in range(Nodenum):
                dis = []
                for idx2 in range(Nodenum):
                    # idx=idx2的时候，必有一个值是0
                    dis.append(math.sqrt(
                        math.pow(nodepos[idx][0] - nodepos[idx2][0], 2) +
                        math.pow(nodepos[idx][1] - nodepos[idx2][1], 2)))
                for idx2 in range(lj+1):
                    idxmin = np.argmin(dis)
                    if idxmin == idx:
                        dis[idxmin] = 1000000
                    else:
                        A[idx][idxmin] = 1
                        dis[idxmin] = 1000000
    else:
        # 使用Delaunay方法获取邻居节点
        Adj = Delaunay(nodepos).simplices
        A = np.zeros((len(TagInf), len(TagInf)))
        for D in Adj:
            A[D[0]][D[1]] = 1
            A[D[1]][D[0]] = 1
            A[D[0]][D[2]] = 1
            A[D[2]][D[0]] = 1
            A[D[1]][D[2]] = 1
            A[D[2]][D[1]] = 1
    for row in A:
        Res.append(np.nonzero(row)[0])
    return Res

def GetCorrelationResultsFromSimilarityMatrix(Mv):
    Mv_ = Mv.copy()
    P = []
    # 获取关联配对的索引
    L = min(Mv_.shape[0], Mv_.shape[1])
    for i in range(L):
        sm = Mv_.max()
        if sm < 0.25:
            break
        else:
            idx = np.unravel_index(Mv_.argmax(), Mv_.shape)
            P.append([idx[0], idx[1]])
            Mv_[idx[0], :] = 0
            Mv_[:, idx[1]] = 0
    return P

def GeTransformationMatrix(Mv, TagInfA, TagInfB):
    Mv_ = Mv.copy()
    P = []
    xyA = []
    xyB = []
    # 获取关联配对的索引
    L = min(Mv_.shape[0], Mv_.shape[1])
    for i in range(L):
        sm = Mv_.max()
        if sm < 0.25:
            break
        else:
            idx = np.unravel_index(Mv_.argmax(), Mv_.shape)
            P.append([idx[0], idx[1]])
            Mv_[idx[0], :] = 0
            Mv_[:, idx[1]] = 0
            xyA = xyA + [[TagInfA[idx[0]][1], TagInfA[idx[0]][2]]]
            xyB = xyB + [[TagInfB[idx[1]][1], TagInfB[idx[1]][2]]]
    xyA = np.matrix(xyA)
    xyB = np.matrix(xyB)

    # 计算单应矩阵，xyA到xyB
    H, state = cv2.findHomography(xyA, xyB, cv2.RANSAC, 25.0)
    return H, P

def NeighborhoodSimilarity1(Mv, H, idxA, idxB, NeigA_, NeigB_, TagInfA, TagInfB):

    CenApos = [TagInfA[idxA][1], TagInfA[idxA][2]]
    CenBpos = [TagInfB[idxB][1], TagInfB[idxB][2]]
    NeigApos = [[TagInfA[idx][1], TagInfA[idx][2]] for idx in NeigA_]
    NeigBpos = [[TagInfB[idx][1], TagInfB[idx][2]] for idx in NeigB_]

    CenApos2B = CalculateThCoordinate([[CenApos[0]], [CenApos[1]], [1.0]], H)
    NeigApos2B = [CalculateThCoordinate([[D[0]], [D[1]], [1.0]], H) for D in NeigApos]

    NeigApos2B_ = [[D[0]-CenApos2B[0], D[1]-CenApos2B[1]] for D in NeigApos2B]
    NeigBpos_ = [[D[0]-CenBpos[0], D[1]-CenBpos[1]] for D in NeigBpos]
    P = []
    Data = []
    for Da in NeigApos2B_:
        dis = []
        for Db in NeigBpos_:
            dis.append( math.pow(Da[0] - Da[0], 2) +
                        math.pow(Db[1] - Db[1], 2))
        Data.append(dis)
    Data = np.matrix(Data)
    for i in range(min(len(NeigA_), len(NeigB_))):
        idx = np.unravel_index(Data.argmin(), Data.shape)
        P.append(idx)
        Data[idx[0], :] = 10000000
        Data[:, idx[1]] = 10000000
    # print(P)
    m = Mv[idxA][idxB] * len(P)
    for p in P:
        m += Mv[p[0]][p[1]]
    m = m / len(P)
    return m

#
def NeighborhoodSimilarity2(Mv, idxA, idxB, NeigA_, NeigB_, TagInfA, TagInfB):

    CenApos = [TagInfA[idxA][1], TagInfA[idxA][2]]
    CenBpos = [TagInfB[idxB][1], TagInfB[idxB][2]]
    NeigApos = [[TagInfA[idx][1], TagInfA[idx][2]] for idx in NeigA_]
    NeigBpos = [[TagInfB[idx][1], TagInfB[idx][2]] for idx in NeigB_]

    NeigApos_ = [[D[0]-CenApos[0], D[1]-CenApos[1]] for D in NeigApos]
    NeigBpos_ = [[D[0]-CenBpos[0], D[1]-CenBpos[1]] for D in NeigBpos]

    NeigAazim = [AzimuthAngle(0, 0, D[0], D[1]) for D in NeigApos_]
    NeigBazim = [AzimuthAngle(0, 0, D[0], D[1]) for D in NeigBpos_]
    # print(NeigBazim)
    NeigBazim_ = AzimuthalRotation(NeigBazim, min(NeigBazim))
    m = 0
    for D in NeigAazim:
        NeigAazim_ = AzimuthalRotation(NeigAazim, D)
        P = GetAzimuthCorrelationsAndDeviations(NeigAazim_, NeigBazim_)

        m_ = CalculatingSimilarity(P, Mv, NeigA_, NeigB_)
        m = m if m > m_ else m_

    m += Mv[idxA][idxB]
    return m / 2

def SimilarityAggregationNeighbors(Mv, NeigA, NeigB, TagInfA, TagInfB):
    # print(Mv.shape[0], Mv.shape[1])
    # print(len(TagInfA), len(TagInfB))
    rows_n = len(TagInfA)
    cols_n = len(TagInfB)
    Mu = np.zeros((rows_n, cols_n))
    for idxA in range(rows_n):
        NeigA_ = NeigA[idxA]
        for idxB in range(cols_n):
            NeigB_ = NeigB[idxB]
            Mu[idxA][idxB] = NeighborhoodSimilarity2(Mv, idxA, idxB, NeigA_, NeigB_, TagInfA, TagInfB)
    return Mu

def ShowResult(Gimage, P, TagInfA, TagInfB, ImageName = None):
    Gimage_ = Gimage.copy()
    PY = int(0.5*(Gimage_.shape[1]))
    thickness = 12
    lineType = 100
    for D in P:
        [idxA, idxB] = D
        # 每一对目标都生成不同的颜色
        Gcolor = (randint(0, 255), randint(0, 255), randint(0, 255))
        # 左边图片的点
        ptStart = (int(TagInfA[idxA][1]), int(TagInfA[idxA][2]))
        # 右边图片的点
        ptEnd = (PY + int(TagInfB[idxB][1]), int(TagInfB[idxB][2]))
        # 画点
        # cv2.circle(Gimage_, ptStart, 50, Gcolor, thickness=-1)
        # cv2.circle(Gimage_, ptEnd, 50, Gcolor, thickness=-1)
        # 画线
        cv2.line(Gimage_, ptStart, ptEnd, Gcolor, thickness, lineType)

    if ImageName:
        cv2.namedWindow(ImageName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ImageName, 1920, 540)
        cv2.imshow(ImageName, Gimage_)

    return Gimage_

def CorrectCorrelationRate(P_all, P_res, La, Lb):
    res = []
    for D in P_all:
        Ta = np.zeros((La))
        Tb = np.zeros((Lb))
        for d in D:
            Ta[d[0]] = 1
            Tb[d[1]] = 1
        # 应该被关联的目标
        AllAssociNum = len(P_res)
        # 不应该被关联的目标
        AllDisassociNum = [La-AllAssociNum, Lb-AllAssociNum]
        # 结果中关联的数量
        ResAssociNum = len(D)
        # 结果中正确的关联
        ResRight = 0
        # 结果中存在不存在关联的目标的关联
        ResUndeservedAssociNum = 0
        for d in D:
            if d in P_res:
                ResRight += 1
            else:
                if Ta[d[0]] == 0 or Tb[d[1]] == 0:
                    ResUndeservedAssociNum += 1
        res.append([AllAssociNum, AllDisassociNum[0], AllDisassociNum[1], ResAssociNum, ResRight, ResUndeservedAssociNum])
    return res

if __name__ == '__main__':
    thea = AzimuthAngle(0, 0, -1, 1)
    print(thea)

