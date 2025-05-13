#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 4:32
# @Author : ZM7
# @File : new_main
# @Software: PyCharm

import datetime
import torch
from sys import exit
import pandas as pd
import numpy as np
from DGSR import DGSR, collate, collate_test #, FocalLoss
from dgl import load_graphs
import pickle
from utils import myFloder
import warnings
import argparse
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger, eval_metric_with_labels
import json
import multiprocessing
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sample', help='data name: sample')
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
    parser.add_argument('--user_update', default='rnn')
    parser.add_argument('--item_update', default='rnn')
    parser.add_argument('--user_long', default='orgat')
    parser.add_argument('--item_long', default='orgat')
    parser.add_argument('--user_short', default='att')
    parser.add_argument('--item_short', default='att')
    parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
    parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
    parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
    parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
    parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')
    parser.add_argument('--k_hop', type=int, default=2, help='sub-graph size')
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--last_item', action='store_true',default=True, help='aggregate last item')
    parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
    parser.add_argument("--val", action='store_true', default=True)
    parser.add_argument("--model_record", action='store_true', default=True, help='record model')

    opt = parser.parse_args()
    args, extras = parser.parse_known_args()
    device = torch.device('cuda:0')
    print(opt)

    if opt.record:
        log_file = f'results/{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                f'_layer_{opt.layer_num}_l2_{opt.l2}'
        mkdir_if_not_exist(log_file)
        sys.stdout = Logger(log_file)
        print(f'Logging to {log_file}')
    if opt.model_record:
        model_file = f'{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                f'_layer_{opt.layer_num}_l2_{opt.l2}'

    
    # loading data
    data = pd.read_csv('./Data/' + opt.data + '.csv')
    # data = data[data['user_id'] < 2600]
    user = data['user_id'].unique()
    item = data['item_id'].unique()

    # # Get frequency of each item and compute weights for cross-entropy loss
    # item_frequencies = data['item_id'].value_counts().reindex(item, fill_value=0)
    # weights = 1.0 / item_frequencies.values
    # weights /= weights.sum() * len(weights)
    # ce_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    user_num = len(user)
    item_num = len(item)
    train_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/' # original code
    # train_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/retrain/'
    test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/' # original code
    # test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/new_test/'
    val_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
    train_set = myFloder(train_root, load_graphs)
    test_set = myFloder(test_root, load_graphs)
    if opt.val:
        val_set = myFloder(val_root, load_graphs)

    print('train number:', train_set.size)
    print('test number:', test_set.size)
    print('user number:', user_num)
    print('item number:', item_num)
    f = open(opt.data+'_neg', 'rb')
    data_neg = pickle.load(f) # 用于评估测试集 translation: for evaluating the test set
    train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=4)
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=0) # original code
    # test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=collate, pin_memory=True, num_workers=0, drop_last=True) # used collate instead of collate_test
    if opt.val:
        val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=0) # original code
        # val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=collate, pin_memory=True, num_workers=0, drop_last=True) # used collate instead of collate_test


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Initialize model
    model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
                user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, user_long=opt.user_long, user_short=opt.user_short,
                item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update, item_update=opt.item_update, last_item=opt.last_item,
                layer_num=opt.layer_num).cuda()

    # Print the number of parameters
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.CrossEntropyLoss(weight=ce_weights)
    # loss_func = FocalLoss(alpha=0.5, gamma=5)
    scaler = GradScaler()
    best_result = [0, 0, 0, 0, 0, 0]   # hit5,hit10,hit20,mrr5,mrr10,mrr20
    best_epoch = [0, 0, 0, 0, 0, 0]
    stop_num = 0

    # # model.load_state_dict(torch.load('save_models/vw_state_dict_300.pkl'))
    # state_dict = torch.load(r"C:\Users\omar9\Desktop\thesis\github_repo\dgsr_vw\save_models\vw_2500_seen_label_ba_50_G_4_dim_50_ulong_orgat_ilong_orgat_US_att_IS_att_La_True_UM_50_IM_50_K_2_layer_3_l2_0.0001.pkl")
    # user_embeddings_tensor = state_dict['user_embedding.weight']

    # num_pretrained_users = user_embeddings_tensor.shape[0]
    # new_user_ids = list(range(num_pretrained_users, len(user)))

    # # Randomly initialize new user embeddings
    # new_user_embeddings = nn.Embedding(len(new_user_ids), user_embeddings_tensor.shape[1]).to(device).weight

    # # Concatenate new_user_embeddings_tensor to user_embeddings_tensor
    # user_embeddings_tensor = torch.cat([user_embeddings_tensor, new_user_embeddings], dim=0)

    # # Save the updated user embeddings to the model state dictionary
    # state_dict['user_embedding.weight'] = user_embeddings_tensor
    # model.load_state_dict(state_dict)

    # Initialize an empty dictionary to accumulate embeddings
    # all_user_profiles = {}
    # all_unified_embeddings = {}

    for epoch in range(opt.epoch):
        stop = True
        epoch_loss = 0
        iter = 0
        print('start training: ', datetime.datetime.now())
        model.train()
        for user, batch_graph, label, last_item in train_data:
            iter += 1
            with autocast():
                score, user_profiles_dict, unified_embeddings_dict = model(batch_graph.to(device), user.to(device), last_item.to(device), is_training=True, return_user_profiles=True)
                loss = loss_func(score, label.to(device))
            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            # # Convert int64 keys to strings and update the accumulated dictionary
            # user_profiles_dict = {str(k): v for k, v in user_profiles_dict.items()}
            # unified_embeddings_dict = {str(k): v for k, v in unified_embeddings_dict.items()}
            # all_unified_embeddings.update(unified_embeddings_dict)
            # all_user_profiles.update(user_profiles_dict)

            if iter % 400 == 0:
            #     with open('user_profiles_750_focal_loss.json', 'w') as f:
            #         json.dump(all_user_profiles, f)
            #     with open('unified_embeddings_750_focal_loss.json', 'w') as f:
            #         json.dump(all_unified_embeddings, f)
                print('Iter {}, loss {:.4f}'.format(iter, epoch_loss/iter), datetime.datetime.now())
        
        epoch_loss /= iter
        model.eval()
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), '=============================================')

        # val
        if opt.val:
            print('start validation: ', datetime.datetime.now())
            val_loss_all, top_val, all_label = [], [], []
            with torch.no_grad():
                for user, batch_graph, label, last_item, neg_tar in val_data: # original code
                # for user, batch_graph, label, last_item in test_data:
                    score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)
                    
                    # # Ranking all items for each user instead of using negative samples
                    # neg_tar_list = list(range(0, 920))
                    # neg_tar = torch.tensor([neg_tar_list] * 50).to(device)  # Creates a 50x920 tensor
                    # score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=neg_tar, is_training=False)

                    val_loss = loss_func(score, label.cuda())
                    val_loss_all.append(val_loss.item())
                    top_val.append(top.detach().cpu().numpy())
                    all_label.append(label.numpy())
                recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_val, all_label)
                print('train_loss:%.4f\tval_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                    '\tNDGG10@10:%.4f\tNDGG@20:%.4f' %
                    (epoch_loss, np.mean(val_loss_all), recall5, recall10, recall20, ndgg5, ndgg10, ndgg20))

        # test
        print('start predicting: ', datetime.datetime.now())
        all_top, all_label, all_length = [], [], []
        iter = 0
        all_loss = []
        with torch.no_grad():
            for user, batch_graph, label, last_item, neg_tar in test_data: # original code
            # for user, batch_graph, label, last_item in test_data:
                iter+=1
                score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=torch.cat([label.unsqueeze(1), neg_tar],-1).to(device), is_training=False)

                # # Ranking all items for each user instead of using negative samples
                # neg_tar_list = list(range(0, 920))
                # neg_tar = torch.tensor([neg_tar_list] * 50).to(device)  # Creates a 50x920 tensor
                # score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=neg_tar, is_training=False)

                test_loss = loss_func(score, label.cuda())
                all_loss.append(test_loss.item())
                
                all_top.append(top.detach().cpu().numpy())
                all_label.append(label.numpy())

                if iter % 200 == 0:
                    print('Iter {}, test_loss {:.4f}'.format(iter, np.mean(all_loss)), datetime.datetime.now())
            recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(all_top, all_label)

            if recall5 > best_result[0]:
                best_result[0] = recall5
                best_epoch[0] = epoch
                stop = False
            if recall10 > best_result[1]:
                if opt.model_record:
                    torch.save(model.state_dict(), 'save_models/'+ model_file + '.pkl')
                best_result[1] = recall10
                best_epoch[1] = epoch
                stop = False
            if recall20 > best_result[2]:
                best_result[2] = recall20
                best_epoch[2] = epoch
                stop = False
                # ------select Mrr------------------
            if ndgg5 > best_result[3]:
                best_result[3] = ndgg5
                best_epoch[3] = epoch
                stop = False
            if ndgg10 > best_result[4]:
                best_result[4] = ndgg10
                best_epoch[4] = epoch
                stop = False
            if ndgg20 > best_result[5]:
                best_result[5] = ndgg20
                best_epoch[5] = epoch
                stop = False
            if stop:
                stop_num += 1
            else:
                stop_num = 0
            print('train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                '\tNDGG10@10:%.4f\tNDGG@20:%.4f\tEpoch:%d,%d,%d,%d,%d,%d' %
                (epoch_loss, np.mean(all_loss), best_result[0], best_result[1], best_result[2], best_result[3],
                best_result[4], best_result[5], best_epoch[0], best_epoch[1],
                best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))
