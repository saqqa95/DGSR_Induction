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
from DGSR import DGSR, collate, collate_test
import dgl
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


warnings.filterwarnings('ignore')

if __name__ == '__main__':
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
    parser.add_argument('--last_item', action='store_true',default=True, help='aggreate last item')
    parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
    parser.add_argument("--val", action='store_true', default=False)
    parser.add_argument("--model_record", action='store_true', default=False, help='record model')

    opt = parser.parse_args()
    args, extras = parser.parse_known_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    print(f"Running on device: {device}")
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

    # Optionl: Load a specific dataset:
    # data_name = "vw_3900_unseen_label_inference"
    # data = pd.read_csv('./Data/' + data_name + '.csv')
    # data = pd.read_csv(r"C:\Users\omar9\Desktop\thesis\DGSR-master\Data\Vectorworks_307.csv")

    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item)
    train_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
    test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/new_test/'
    # test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
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
    data_neg = pickle.load(f) # 用于评估测试集
    train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=0)
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=0, drop_last=True)
    if opt.val:
        val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=0)

    # 初始化模型
    model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
                user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, user_long=opt.user_long, user_short=opt.user_short,
                item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update, item_update=opt.item_update, last_item=opt.last_item,
                layer_num=opt.layer_num).cuda()
    # print(f'Length of trianing data: {len(train_data)}')       # flag added by omar
    # print(f'Length of testing data: {len(test_data)}')        # flag added by omar
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    # loss_func = nn.CrossEntropyLoss()
    # best_result = [0, 0, 0, 0, 0, 0]   # hit5,hit10,hit20,mrr5,mrr10,mrr20
    # best_epoch = [0, 0, 0, 0, 0, 0]
    # stop_num = 0

    # Load the model state dictionary
    state_dict = torch.load(r"C:\Users\omar9\Desktop\thesis\github_repo\dgsr_vw\save_models\train_10k_ba_50_G_4_dim_50_ulong_orgat_ilong_orgat_US_att_IS_att_La_True_UM_50_IM_50_K_2_layer_3_l2_0.0001.pkl")

    user_embeddings_tensor = state_dict['user_embedding.weight']
    item_embeddings_tensor = state_dict['item_embedding.weight']

    print('User embeddings tensor shape:', user_embeddings_tensor.shape)

    first_user_embedding = user_embeddings_tensor[0].cpu().numpy()

    default_embedding = user_embeddings_tensor.mean(dim=0).cpu().numpy()

    num_pretrained_users = user_embeddings_tensor.shape[0]
    # num_pretrained_users = 2500

    pretrained_user_ids = list(range(num_pretrained_users))
    new_user_ids = list(range(num_pretrained_users, user_num))
   
    # Define a function to calculate Jaccard similarity between two sequences
    def jaccard_similarity(seq1, seq2):
        set1, set2 = set(seq1), set(seq2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 0
        return intersection / union

    # Get active sequence for each new user, calculate similarity with pretrained users, and get weighted average of the similar user embeddings
    similarity_dict = {}
    new_user_embeddings = []
    test_length = 100

    for active_user in new_user_ids:
        active_user_seq = data[data['user_id'] == active_user]['item_id'].tolist()
        # active_user_seq = active_user_seq[:-1]  # Remove the last item from the sequence

        # Limit the sequence to the most recent test_length items if it is longer
        if len(active_user_seq) > test_length:
            active_user_seq_train = active_user_seq[-test_length:]
            active_user_seq = active_user_seq_train[-(test_length-1):]
            # target_item = active_user_seq_train[-1]
        else:
            active_user_seq_train = active_user_seq
            # target_item = active_user_seq_train[-1]
        
        for user_id in pretrained_user_ids:
            pretrained_user_seq = data[data['user_id'] == user_id]['item_id'].tolist()
            pretrained_user_seq = pretrained_user_seq[:-1]
            similarity_dict[user_id] = jaccard_similarity(active_user_seq, pretrained_user_seq)
        
        # Get similar users and embeddings    
        similar_users = [user_id for user_id, similarity in similarity_dict.items() if similarity > 0]
        similar_users_embeddings = np.array([user_embeddings_tensor[user_id].cpu().numpy() for user_id in similar_users])
        similarity_matrix = np.array([similarity_dict[user_id] for user_id in similar_users])

        # similar_users = [user_id for user_id, similarity in similarity_dict.items() if similarity > 0]
        # similar_users_embeddings = torch.stack([user_embeddings_tensor[user_id] for user_id in similar_users])
        # similarity_matrix = torch.tensor([similarity_dict[user_id] for user_id in similar_users], device=similar_users_embeddings.device)

        # Calculate weighted average of similar user embeddings
        if len(similar_users) > 0:
            weighted_average = np.average(similar_users_embeddings, axis=0, weights=similarity_matrix)
            new_user_embeddings.append(weighted_average)
        else:
            new_user_embeddings.append(default_embedding)

        # # Calculate weighted average of similar user embeddings
        # if len(similar_users) > 0:
        #     weighted_average = np.average(similar_users_embeddings, axis=0, weights=similarity_matrix)
        # else:
        #     # Assign default embedding if no similar users found
        #     new_user_embeddings.append(default_embedding)

    #     # if len(similar_users) > 0:
    #     #     similarity_matrix = similarity_matrix / similarity_matrix.sum()  # Normalize weights
    #     #     weighted_average = torch.sum(similar_users_embeddings * similarity_matrix[:, None], dim=0)
    #     #     new_user_embeddings.append(weighted_average.cpu().numpy())
    #     # else:
    #     #     new_user_embeddings.append(default_embedding)
        
    #     print(f"Length of new user embeddings list: {len(new_user_embeddings)}")

    # Convert new_user_embeddings to a tensor
    new_user_embeddings_tensor = torch.tensor(new_user_embeddings).to(device).to(torch.float32)

    # Concatenate new_user_embeddings_tensor to user_embeddings_tensor
    user_embeddings_tensor = torch.cat([user_embeddings_tensor, new_user_embeddings_tensor], dim=0)

    # Save the updated user embeddings to the model state dictionary
    state_dict['user_embedding.weight'] = user_embeddings_tensor

    # print('Updated user embeddings tensor shape:', user_embeddings_tensor.shape)

    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.eval()

    # first_user_embedding_after = user_embeddings_tensor[0].cpu().numpy()

    # if np.array_equal(first_user_embedding, first_user_embedding_after):
    #     print("User embeddings were not changed")
    # else:
    #     print("User embeddings were changed")
    

    print('start predicting: ', datetime.datetime.now())
    all_top, all_label, all_length = [], [], []
    # specific_user_top, specific_user_label = [], []
    iter = 0
    all_loss = []

    with torch.no_grad():
        for user, batch_graph, label, last_item, neg_tar in test_data:
            iter += 1
            # Move data to the correct device
            user = user.to(device)
            batch_graph = batch_graph.to(device)
            last_item = last_item.to(device)
            # neg_tar = torch.cat([label.unsqueeze(1), neg_tar], -1).to(device)

            # Generate predictions
            # score, top = model(batch_graph, user, last_item, neg_tar=neg_tar, is_training=False) # Original line
            # score, top = model(batch_graph, user, last_item, is_training=False)

            # Ranking all items for each user instead of using negative samples
            neg_tar_list = list(range(0, 1146))
            neg_tar = torch.tensor([neg_tar_list] * 50).to(device)  # Creates a 50x1146 tensor
            score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=neg_tar, is_training=False)

            # # Calculate loss
            # test_loss = loss_func(score, label.cuda())
            # all_loss.append(test_loss.item())
            
            # Collect predictions and labels
            all_top.append(top.detach().cpu().numpy())
            # all_top.append(score.detach().cpu().numpy())

            all_label.append(label.numpy())
            
            # Print intermediate results for debugging
            if iter % 200 == 0:
                print('Iter {}, test_loss {:.4f}'.format(iter, np.mean(all_loss)), datetime.datetime.now())
                print('Score:', score.size())
                print('Top:', top.size())
                print('Label:', label)
            

        # Evaluate metrics
        # recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(all_top)
        recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric_with_labels(all_top, all_label)
        # recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_k_indices)
        print('Recall@5:',  recall5)
        print('Recall@10:', recall10)
        print('Recall@20:', recall20)
        print('NDCG@5:',    ndgg5)
        print('NDCG@10:',   ndgg10)
        print('NDCG@20:',   ndgg20)

        # # Evaluate metrics for new users
        # new_recall5, new_recall10, new_recall20, new_ndgg5, new_ndgg10, new_ndgg20 = eval_metric(new_user_top)
        # print('New User Recall@5:',  new_recall5)
        # print('New User Recall@10:', new_recall10)
        # print('New User Recall@20:', new_recall20)
        # print('New User NDGG@5:',    new_ndgg5)
        # print('New User NDGG@10:',   new_ndgg10)
        # print('New User NDGG@20:',   new_ndgg20)

        # # Print top predictions for specific users
        # print('Top predictions for specific users:')
        # for user_id, top_pred in zip(specific_user_ids, specific_user_top):
        #     print(f'User {user_id} - Top: {top_pred}')
