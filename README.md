# SegNet-edg
A Two-way Network for Semantic Segmentation of High-resolution Remote Sensing Images


网络模型基于pytorch实现。


dataset：数据集的存放目录 数据集可以为任意的三通道遥感图像。本模型使用ISPRS Potsdam和ISPRS Vaihingen数据集。链接: https://pan.baidu.com/s/1hR6FaBTvyH3nRSLECAB1Nw 提取码: qf8e



preprocess.py：对数据集进行预处理，生成训练集


processval.py:生成验证集


运行train_seg_edg.py可直接开始训练模型。

