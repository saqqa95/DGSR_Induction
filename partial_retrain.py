import datetime
import torch
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
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger
import json


warnings.filterwarnings('ignore')

# Inference using randomly initialized new user nodes

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
    parser.add_argument('--last_item', action='store_true', help='aggreate last item')
    parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
    parser.add_argument("--val", action='store_true', default=False)
    parser.add_argument("--model_record", action='store_true', default=True, help='record model')

    opt = parser.parse_args()
    args, extras = parser.parse_known_args()
    device = torch.device('cuda:0')
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

    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item)
    train_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
    test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/new_test/'
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
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=0)
    if opt.val:
        val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=0)

    # Load the model state dictionary
    state_dict = torch.load(r"C:\Users\omar9\Desktop\thesis\github_repo\dgsr_vw\save_models\vw_80_600_ba_50_G_4_dim_50_ulong_orgat_ilong_orgat_US_att_IS_att_La_False_UM_50_IM_50_K_2_layer_3_l2_0.0001.pkl")

    user_embeddings_tensor = state_dict['user_embedding.weight']

    num_pretrained_users = user_embeddings_tensor.shape[0]
    new_user_ids = list(range(num_pretrained_users, len(user)))

    # Randomly initialize new user embeddings
    new_user_embeddings = nn.Embedding(len(new_user_ids), user_embeddings_tensor.shape[1]).to(device).weight

    # Concatenate new_user_embeddings_tensor to user_embeddings_tensor
    user_embeddings_tensor = torch.cat([user_embeddings_tensor, new_user_embeddings], dim=0)

    # Save the updated user embeddings to the model state dictionary
    state_dict['user_embedding.weight'] = user_embeddings_tensor
    model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
                user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, user_long=opt.user_long, user_short=opt.user_short,
                item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update, item_update=opt.item_update, last_item=opt.last_item,
                layer_num=opt.layer_num).cuda()
    
    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.eval()

    print('start predicting: ', datetime.datetime.now())
    all_top, all_label, all_length = [], [], []
    iter = 0
    all_loss = []

    with torch.no_grad():
        for user, batch_graph, label, last_item, neg_tar in test_data:
            iter += 1
            # Move data to the correct device
            user = user.to(device)
            batch_graph = batch_graph.to(device)
            last_item = last_item.to(device)
            neg_tar = torch.cat([label.unsqueeze(1), neg_tar], -1).to(device)

            # Generate predictions
            score, top = model(batch_graph, user, last_item, neg_tar=neg_tar, is_training=False)

            # # Calculate loss
            # test_loss = loss_func(score, label.cuda())
            # all_loss.append(test_loss.item())
            
            # Collect predictions and labels
            all_top.append(top.detach().cpu().numpy())
            all_label.append(label.numpy())
            
            # Print intermediate results for debugging
            if iter % 200 == 0:
                print('Iter {}, test_loss {:.4f}'.format(iter, np.mean(all_loss)), datetime.datetime.now())
                print('Score:', score.size())
                print('Top:', top.size())
                print('Label:', label)
            
        # Evaluate metrics
        recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(all_top)
        print('Recall@5:',  recall5)
        print('Recall@10:', recall10)
        print('Recall@20:', recall20)
        print('NDCG@5:',    ndgg5)
        print('NDCG@10:',   ndgg10)
        print('NDCG@20:',   ndgg20)