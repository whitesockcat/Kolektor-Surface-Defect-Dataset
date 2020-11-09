"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os
from transform import *
# from ranDefect import add_noise

def default_loader(id, root):
    img = cv2.imread(root+'/img/'+id)
    mask = cv2.imread(root + '/seg/' + id, cv2.IMREAD_GRAYSCALE)
    
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    return img, mask

class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)