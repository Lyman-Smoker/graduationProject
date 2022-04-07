import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from scipy import stats
import random
import argparse
import time
import seaborn as sns
import pandas as pd
import copy
# dataset
from dataset import SevenPair_all_Dataset, get_video_trans, worker_init_fn
# models
from models.I3D_Backbone import I3D_backbone
from models.Bidir_Attention import Bidir_Attention
from models.MLP import MLP
from models.LN_MLP import LN_MLP
# optimizer
import torch.optim as optim
import traceback
# visualization
from torch.utils.tensorboard import SummaryWriter
import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.autograd import Variable

# constraint the learing of attention module
def diversity_loss_1(attention):
    attention_t = torch.transpose(attention, 1, 2)
    num_features = attention.shape[1]
    # res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).cuda()
    res = (attention @ attention_t) - torch.eye(num_features)
    res = res.view(-1, num_features*num_features)
    return torch.norm(res, p=2, dim=1).sum() / attention.size(0)

def diversity_loss_2(attention):
    attention_t = torch.transpose(attention, 1, 2)
    num_features = attention.shape[1]
    # res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).cuda()
    res = (attention @ attention_t).softmax(dim=-1) - torch.eye(num_features)
    res = res.view(-1, num_features*num_features)
    return torch.norm(res, p=2, dim=1).sum() / attention.size(0)

def diversity_loss_3(attention):
    attention_t = torch.transpose(attention, 1, 2)
    num_features = attention.shape[1]
    # res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).cuda()
    res = (attention @ attention_t)
    dis = torch.norm(attention, 2, 2)
    dis = dis.unsqueeze(-1)
    dis_mask = dis @ dis.transpose(1, 2)
    res = res / dis_mask
    res = res - torch.eye(num_features)
    res = res.view(-1, num_features*num_features)
    return torch.norm(res, p=2, dim=1).sum() / attention.size(0)


# a = [[0.1 for _ in range(10)] for _ in range(10)]
a = [[1,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1],]
a = np.array(a)
a = torch.from_numpy(a)
a = a.unsqueeze(0)
a = a.float()


variable = Variable(a, requires_grad=True)

optimizer = optim.Adam([
        {'params': variable},
    ], lr=0.1)

print('loss: ', diversity_loss_3(a))
# attention = copy.deepcopy(variable.data)
# attention = attention.softmax(dim=-1)
# attention_t = torch.transpose(attention, 1, 2)
# # res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).cuda()
# print('original loss:', diversity_loss_1(attention))

# res = (attention @ attention_t)
sns.heatmap(
    pd.DataFrame(np.round(copy.deepcopy(variable.data)[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
plt.savefig('./V_original.png')
plt.close()
# sns.heatmap(
#     pd.DataFrame(np.round(attention[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
# plt.savefig('./A_original.png')
# # plt.show()
# plt.close()
# sns.heatmap(
#     pd.DataFrame(np.round(res[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
# plt.savefig('./AA_original.png')
# # plt.show()
# plt.close()


# for epoch in range(300):
#     # print('epoch:', epoch)
#     attn = variable.softmax(dim=-1)
#     loss = 10 * diversity_loss_1(attn)
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# attention = copy.deepcopy(variable.data)
# attention = attention.softmax(dim=-1)
# attention_t = torch.transpose(attention, 1, 2)
# # res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).cuda()
# res = (attention @ attention_t)
# sns.heatmap(
#     pd.DataFrame(np.round(copy.deepcopy(variable.data)[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
# plt.savefig('./V_final.png')
# plt.close()
# sns.heatmap(
#     pd.DataFrame(np.round(attention[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
# plt.savefig('./A_final.png')
# # plt.show()
# plt.close()
# sns.heatmap(
#     pd.DataFrame(np.round(res[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
# plt.savefig('./AA_final.png')
# # plt.show()
# plt.close()
# print('final loss:', diversity_loss_3(attention))
