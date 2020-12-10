import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from edge_utils import mask_to_onehot,onehot_to_binary_edges


class Mydataset(Dataset):

    def __init__(self, image_path, label_path):
        super(Mydataset, self).__init__()
        self.image_path = np.array(image_path)
        self.label_path = np.array(label_path)


    def __len__(self):
        return self.image_path.shape[0]

    def __getitem__(self, idx):
        img = cv2.imread(self.image_path[idx], cv2.IMREAD_COLOR)
        label = cv2.imread(self.label_path[idx], cv2.IMREAD_GRAYSCALE)
        edgemap = label
        edgemap = mask_to_onehot(edgemap, 6)
        edgemap = onehot_to_binary_edges(edgemap, 2, 6)
        edgemap = torch.from_numpy(edgemap).float()
        img_tensor = torch.tensor(img)
        label_tensor = torch.tensor(label).long()
        return img_tensor.float(), label_tensor, edgemap
