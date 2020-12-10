import numpy as np
import cv2

# 给标签图上色
def color_predicts(img):

    '''
    给class图上色

    '''

    color = np.ones([img.shape[0], img.shape[1], 3])

    color[img == 0] = [255, 255, 255] #地面，白色，0
    color[img == 1] = [0, 0, 255]     #建筑，蓝色，1
    color[img == 2] = [0, 255, 255]   #低植被，青色，2
    color[img == 3] = [0, 255, 0]     #树，绿色，3
    color[img == 4] = [255, 255, 0]   #车，黄色，4
    color[img == 5] = [255 ,0, 0]   #背景，红色，5


    return color

def edge_predicts(img):

    '''
    给class图上色

    '''

    color = np.ones([img.shape[0], img.shape[1], 3])

    color[img == 0] = [255, 255, 255] #非边缘
    color[img == 1] = [0, 0, 0]     #边缘



    return color
def color_annotation(label_path, output_path):

    '''

    给class图上色

    '''


    color = np.ones([img.shape[0], img.shape[1], 3])

    color[img == 0] = [255, 255, 255] #地面，白色，0
    color[img == 1] = [0, 0, 255]     #建筑，蓝色，1
    color[img == 2] = [0, 255, 255]   #低植被，青色，2
    color[img == 3] = [0, 255, 0]     #树，绿色，3
    color[img == 4] = [255, 255, 0]   #车，黄色，4
    color[img == 5] = [255, 0, 0]   #背景，红色，5


    cv2.imwrite(output_path,color)