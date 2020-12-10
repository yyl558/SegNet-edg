import cv2
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import os

size = 256# 随机窗口采样
def generate_val_dataset(image_num = 800,
                           val_image_path='dataset/val/images/',
                           val_label_path='dataset/val/labels/'):
    '''
    该函数用来生成训练集，切图方法为随机切图采样
    :param image_num: 生成样本的个数
    :param train_image_path: 切图保存样本的地址
    :param train_label_path: 切图保存标签的地址
    :return:
    '''

    # 用来记录所有的子图的数目
    g_count = 1
    images_path = ['dataset/origin/top_potsdam_3_13_RGB.tif']
    labels_path = ['dataset/origin/top_potsdam_3_13_class.png']



    # 每张图片生成子图的个数
    image_each = image_num // len(images_path)
    image_path, label_path = [], []
    for i in tqdm(range(len(images_path))):
        count = 0
        image = cv2.imread(images_path[i])
        label = cv2.imread(labels_path[i], 0)
        X_height, X_width = image.shape[0], image.shape[1]
        while count < image_each:
            random_width = random.randint(0, X_width - size - 1)
            random_height = random.randint(0, X_height - size - 1)
            image_ogi = image[random_height: random_height + size, random_width: random_width + size,:]
            label_ogi = label[random_height: random_height + size, random_width: random_width + size]

            #image_d, label_d = data_augment(image_ogi, label_ogi)
            image_d, label_d = image_ogi, label_ogi
            image_path.append(val_image_path+'%05d.png' % g_count)
            label_path.append(val_label_path+'%05d.png' % g_count)z
            cv2.imwrite((val_image_path+'%05d.png' % g_count), image_d)
            cv2.imwrite((val_label_path+'%05d.png' % g_count), label_d)

            count += 1
            g_count += 1
    df = pd.DataFrame({'image':image_path, 'label':label_path})
    df.to_csv('dataset/path_val_list.csv', index=False)



if __name__ == '__main__':
    if not os.path.exists('dataset/val/images'): os.mkdir('dataset/val/images')
    if not os.path.exists('dataset/val/labels'): os.mkdir('dataset/val/labels')
    generate_val_dataset()
