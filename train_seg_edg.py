from dataset_mynet import Mydataset
from torch.utils.data import DataLoader
import pandas as pd
from seg_edge import SegNet_Edge
import torch.nn as nn
import torch
import loss
from metircs import Evaluator
import os
import numpy as np
import cv2
from color_utils import color_predicts, edge_predicts


def main():


    data_path_df_train = pd.read_csv('dataset/path_list.csv')
    data_path_df_val = pd.read_csv('dataset/path_val_list.csv')
    log_path = 'logs/'

    data_train = Mydataset(image_path=data_path_df_train['image'].values, label_path=data_path_df_train['label'].values)
    data_val = Mydataset(image_path=data_path_df_val['image'].values, label_path=data_path_df_val['label'].values)
    data_train_loader = DataLoader(dataset=data_train, batch_size=16)
    data_val_loader = DataLoader(dataset=data_val, batch_size=16)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.device_count())

    model = SegNet_Edge()
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = loss.JointEdgeSegLoss()
    evaluator = Evaluator(6)
    count = 1

    for i in range(20000):
        print(i)
        model.train()
        for _, data in enumerate(data_train_loader):
            img, label, edge = data
            img, label, edge = img.cuda(), label.cuda(), edge.cuda()

            img = img.permute(0, 3, 1, 2)
            seg_out, edge_out = model(img)

            optimizer.zero_grad()
            loss_all = criterion((seg_out, edge_out), (label, edge))
            main_loss = None
            main_loss = loss_all['seg_loss']
            main_loss += loss_all['edge_loss']
            main_loss += loss_all['att_loss']
            main_loss += loss_all['dual_loss']
            main_loss.backward()
            optimizer.step()

        if (i in [2, 50, 100, 200]) or (i > 0 and i % 20 == 0):
            model.eval()
            evaluator.reset()
            for j, val_data in enumerate(data_val_loader):
                img_val, label_val, edge_val = val_data
                img_val, label_val = img_val.cuda(), label_val.cuda()
                img_val = img_val.permute(0, 3, 1, 2)

                with torch.no_grad():
                    seg_predict, edge_predict = model(img_val)

                seg_predict = seg_predict.data.cpu().numpy()
                edge_predict = edge_predict.data.cpu().numpy()
                label_val = label_val.cpu().numpy()
                seg_predict = np.argmax(seg_predict, 1)

                evaluator.add_batch(label_val, seg_predict)


            acc = evaluator.Pixel_Accuracy_Class()
            mIou = evaluator.Mean_Intersection_over_Union()

            precision = evaluator.Precision()
            fscore = evaluator.F1score()

            f = open('logs/result_edg_v_50.txt', 'a')
            f.write("======================%d======================\n" % i)
            f.write('Impervious surfaces iou %.4f\n' % mIou[0])
            f.write('Building iou %.4f\n' % mIou[1])
            f.write('Low vegetation iou %.4f\n' % mIou[2])
            f.write('Tree iou %.4f\n' % mIou[3])
            f.write('Car iou %.4f\n' % mIou[4])
            f.write('background iou %.4f\n' % mIou[5])

            f.write('Impervious surfaces acc %.4f\n' % acc[0])
            f.write('Building acc %.4f\n' % acc[1])
            f.write('Low vegetation acc %.4f\n' % acc[2])
            f.write('Tree acc %.4f\n' % acc[3])
            f.write('Car acc %.4f\n' % acc[4])
            f.write('background acc %.4f\n' % acc[5])

            f.write('Impervious surfaces pre %.4f\n' % precision[0])
            f.write('Building pre %.4f\n' % precision[1])
            f.write('Low vegetation pre %.4f\n' % precision[2])
            f.write('Tree pre %.4f\n' % precision[3])
            f.write('Car pre %.4f\n' % precision[4])
            f.write('backgroundpre %.4f\n' % precision[5])

            f.write('Impervious surfaces F1 %.4f\n' % fscore[0])
            f.write('Building F1 %.4f\n' % fscore[1])
            f.write('Low vegetation F1 %.4f\n' % fscore[2])
            f.write('Tree F1 %.4f\n' % fscore[3])
            f.write('Car F1 %.4f\n' % fscore[4])
            f.write('background F1 %.4f\n' % fscore[5])

            miou = np.nanmean(mIou)
            macc = np.nanmean(acc)
            mpre = np.nanmean(precision)
            mf1 = np.nanmean(fscore)

            f.write('miou: %.4f\n' % miou)
            f.write('macc: %.4f\n' % macc)
            f.write('mpre: %.4f\n' % mpre)
            f.write('mf1: %.4f\n' % mf1)

            f.close()

            print("======================%d======================" % i)

            print('地面iou %.4f' % mIou[0])
            print('建筑 %.4f' % mIou[1])
            print('低植被 %.4f' % mIou[2])
            print('树 %.4f' % mIou[3])
            print('车 %.4f' % mIou[4])
            print('背景 %.4f' % mIou[5])

            print('miou: %.4f' % miou)
            print('acc: %.4f' % macc)


if __name__ == '__main__':
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    main()








































