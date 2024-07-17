# FTVF: 对代码和文件夹的说明

## 代码介绍

main.py    主函数文件

FunctionLib.py  实现算法的一些功能函数

TargetDetection.py  调用yolov5文件夹中的模块，识别并返回图像中的目标的像素坐标

SiameseCNNnet.py    构建孪生图神经网络模型

SiameseCNNtrain.py  训练孪生神经网络模型，这个代码搅乱，没有整理

SiameseCNNpred.py   训练孪生神经网络之后用于测试网络

其他介绍查看具体代码

## data存放的是需要读入的数据

其中，

data是保存输入数据的文件夹;

ImgA和ImgB是存放两架无人机图片的文件夹，图片格式为.jpg;

model是存放孪生神经网络权重的文件夹

result是存放结果的文件夹;

yolov5是整体导入的模块，里面怎么是实现的我不知道，都是在TargetDetection.py里面调用。

