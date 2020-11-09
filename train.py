import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
from tqdm import tqdm
from time import time
from networks.ternausnet import UNet11, AlbuNet, UNet16
from networks.tiramisu import FCDenseNet57, FCDenseNet67, FCDenseNet103
from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet18, LinkNet34, LinkNet50
from networks.dinknet import DinkNet18, DinkNet18_less_pool
from networks.dinknet import DinkNet34, DinkNet34_less_pool
from networks.dinknet import DinkNet50, DinkNet101
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
###########################################
#change NAME and solver
NAME = 'pretrained'
solver = MyFrame(LinkNet18, dice_bce_loss, 1e-4)
BATCHSIZE_PER_CARD = 16
###########################################
SHAPE = (128,128)
ROOT = 'path/to/img/'
trainlist = os.listdir(ROOT+'img/')
trainlist = [name for name in trainlist]

batchsize = BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)

logfolder = 'logs/'
if not os.path.exists(logfolder):
    os.makedirs(logfolder)
mylog = open(logfolder+NAME+'.log','w')
tic = time()
no_optim = 0
total_epoch = 10
train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss.item()
    train_epoch_loss /= len(data_loader_iter)
    print('********', end="", file=mylog)
    print('epoch:',epoch,'    time:',int(time()-tic), file=mylog)
    print('train_loss:',train_epoch_loss, file=mylog)
    print('SHAPE:',SHAPE, file = mylog)
    print('********')
    print ('epoch:',epoch,'    time:',int(time()-tic))
    print ('train_loss:',train_epoch_loss)
    print ('SHAPE:',SHAPE)

    if(epoch%20 == 0 and epoch!=0):
        solver.save('weights/'+NAME+'/'+NAME+str(train_epoch_loss)+'.th')

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+NAME+'.th')
    if no_optim > 20:
        print('early stop at %d epoch' % epoch , file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 10:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(0.8, factor = True, mylog = mylog)
    mylog.flush()
    
print('Finish!', file=mylog)
print ('Finish!')
mylog.close()