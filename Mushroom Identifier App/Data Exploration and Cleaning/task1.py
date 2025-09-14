"""
# IFN680 project 1 Individual Report
Student Name: Le Khanh Linh Pham
Student ID: n10364960

Task 1 - dataloader

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm

def loadData(filedir='trainval', batch_size=32):
    imagenet_means = (0.485, 0.456, 0.406)
    imagenet_stds = (0.229, 0.224, 0.225)
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize(imagenet_means, imagenet_stds)])
    
    trainval_dataset = torchvision.datasets.ImageFolder('trainval', transform = transform)
    
    train_portion = 0.9
    val_portion = 0.1
    
    all_idxes = np.arange(len(trainval_dataset))
    all_targets = trainval_dataset.targets
    
    train_idx, val_idx = train_test_split(all_idxes, train_size=train_portion, stratify = all_targets)
    # print(train_idx)
    # print(val_idx)
    
    train_dataset = torch.utils.data.Subset(trainval_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(trainval_dataset, val_idx)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers = 1)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers = 1)

    return train_dataset, val_dataset, train_dataloader, val_dataloader