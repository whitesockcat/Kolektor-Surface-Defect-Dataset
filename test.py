import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tqdm import tqdm
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import time
from networks.ternausnet import UNet11, AlbuNet, UNet16
from networks.tiramisu import FCDenseNet57, FCDenseNet67, FCDenseNet103
from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet18, LinkNet18, LinkNet50
from networks.dinknet import DinkNet18, DinkNet18_less_pool
from networks.dinknet import DinkNet34, DinkNet34_less_pool
from networks.dinknet import DinkNet50, DinkNet101

BATCHSIZE_PER_CARD = 2

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def test_one_img_from_path(self, path, evalmode = True):
        self.net.eval()
        self.test_one_img_from_path_4(path)

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2
    
    def load(self, path):
        self.net.load_state_dict(torch.load(path))

threshold = .5      
source = 'path/to/img'
val = os.listdir(source)
solver = TTAFrame(LinkNet18)
solver.load('weights/pretrained.th')
target = 'submits/'
if not os.path.exists(target):
    os.mkdir(target)
    
for i,name in enumerate(tqdm(val)):
    mask = solver.test_one_img_from_path(source+name)

    mask[mask> threshold] = 255
    mask[mask<=threshold] = 0
    cv2.imwrite(target+name,mask.astype(np.uint8))

