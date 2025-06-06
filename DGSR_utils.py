#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/19 10:54
# @Author : ZM7
# @File : DGSR_utils
# @Software: PyCharm

import numpy as np
import sys
import torch

def eval_metric(all_top, random_rank=True):
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    # data_l = np.zeros((100, 7))
    for index in range(len(all_top)):
        prediction = (-all_top[index]).argsort(1).argsort(1)
        predictions = prediction[:, 0]
        for i, rank in enumerate(predictions):
            # data_l[per_length[i], 6] += 1
            if rank < 20:
                ndgg20.append(1 / np.log2(rank + 2))
                recall20.append(1)
            else:
                ndgg20.append(0)
                recall20.append(0)
            if rank < 10:
                ndgg10.append(1 / np.log2(rank + 2))
                recall10.append(1)
            else:
                ndgg10.append(0)
                recall10.append(0)
            if rank < 5:
                ndgg5.append(1 / np.log2(rank + 2))
                recall5.append(1)
            else:
                ndgg5.append(0)
                recall5.append(0)
    return np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20)

def eval_metric_with_labels(all_top, all_label):
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    
    for batch_idx in range(len(all_top)):
        # Get predictions and labels for current batch
        prediction = (-all_top[batch_idx]).argsort(1).argsort(1)  # Full ranking matrix
        batch_labels = all_label[batch_idx]  # Labels for current batch
        
        # For each user in the batch
        for user_idx, true_label in enumerate(batch_labels):
            # Get rank of the true item for this user
            rank = prediction[user_idx, true_label]
            
            # Calculate metrics based on rank
            if rank < 20:
                ndgg20.append(1 / np.log2(rank + 2))
                recall20.append(1)
            else:
                ndgg20.append(0)
                recall20.append(0)
                
            if rank < 10:
                ndgg10.append(1 / np.log2(rank + 2))
                recall10.append(1)
            else:
                ndgg10.append(0)
                recall10.append(0)
                
            if rank < 5:
                ndgg5.append(1 / np.log2(rank + 2))
                recall5.append(1)
            else:
                ndgg5.append(0)
                recall5.append(0)
    
    return (np.mean(recall5), np.mean(recall10), np.mean(recall20), 
            np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20))

# def eval_metric(all_top, random_rank=True):
#     recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    
#     for index in range(len(all_top)):
#         # Convert all_top[index] to numpy array and check shape
#         prediction_tensor = all_top[index]
#         if isinstance(prediction_tensor, torch.Tensor):
#             prediction_tensor = prediction_tensor.cpu().numpy()

#         # Ensure prediction is 2D
#         if prediction_tensor.ndim == 1:
#             prediction_tensor = prediction_tensor[np.newaxis, :]  # Expand to 2D if 1D

#         prediction = (-prediction_tensor).argsort(axis=1).argsort(axis=1)
#         predictions = prediction[:, 0]  # Get ranks for the target

#         for rank in predictions:
#             if rank < 20:
#                 ndgg20.append(1 / np.log2(rank + 2))
#                 recall20.append(1)
#             else:
#                 ndgg20.append(0)
#                 recall20.append(0)
#             if rank < 10:
#                 ndgg10.append(1 / np.log2(rank + 2))
#                 recall10.append(1)
#             else:
#                 ndgg10.append(0)
#                 recall10.append(0)
#             if rank < 5:
#                 ndgg5.append(1 / np.log2(rank + 2))
#                 recall5.append(1)
#             else:
#                 ndgg5.append(0)
#                 recall5.append(0)

#     return np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20)




def mkdir_if_not_exist(file_name):
    import os
    import shutil

    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


class Logger(object):
    """
    这个类的目的是尽可能不改变原始代码的情况下, 使得程序的输出同时打印在控制台和保存在文件中
    用法: 只需在程序中加入一行 `sys.stdout = Logger(log_file_path)` 即可
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass