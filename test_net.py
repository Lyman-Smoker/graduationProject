import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from scipy import stats
import random
from dataset import SevenPair_all_Dataset, get_video_trans, worker_init_fn
# models
from models.I3D_Backbone import I3D_backbone
from models.Bidir_Attention import Bidir_Attention
from models.MLP import MLP
from models.LN_MLP import LN_MLP

# fix BatchNorm
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


# load checkpoint
def load_checkpoint(base_model, bidir_attention, ln_mlp, regressor):
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    print(state_dict.keys())

    # parameter resume of base models
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)
    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)
    bidir_attention_ckpt = {k.replace("module.", ""): v for k, v in state_dict['bidir_attention'].items()}
    bidir_attention.load_state_dict(bidir_attention_ckpt)
    ln_mlp_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ln_mlp'].items()}
    ln_mlp.load_state_dict(ln_mlp_ckpt)

    # parameter
    start_epoch = state_dict['epoch']
    epoch_best = state_dict['epoch']
    rho_best = state_dict['rho']
    RL2_min = state_dict['RL2']

    return start_epoch, epoch_best, rho_best, RL2_min


ckpt_path = './ckpt/coat-mlp-sync-10m-div/core-mlp-sync-10m-div_0.9024_0.0360@_199.pth'
class_idx_list = [6]
num_exemplars = 10
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# parameter setting
start_epoch = 0
global epoch_best, rho_best, L2_min, RL2_min
epoch_best = 0
rho_best = 0
L2_min = 1000
RL2_min = 1000
mask = False
vis = True


# load models
base_model = I3D_backbone(I3D_class=400)
bidir_attention = Bidir_Attention(dim=1024, mask=mask, return_attn=vis)
ln_mlp = LN_MLP(2048, use_ln=False, use_mlp=True)
regressor = MLP(in_dim=2049)

# # optimizer & scheduler
# optimizer = optim.Adam([
#     {'params': base_model.parameters(), 'lr':0.001 * 0.1},
#     {'params': bidir_attention.parameters()},
#     {'params': ln_mlp.parameters()},
#     {'params': regressor.parameters()}
#     ], lr=0.001, weight_decay=0.001)
start_epoch, epoch_best, rho_best, RL2_min = load_checkpoint(base_model, bidir_attention, ln_mlp, regressor)
print('resume ckpts @ %d epoch( rho = %.4f, RL2 = %.4f)' % (start_epoch, rho_best, RL2_min))

# CUDA & DP
base_model = base_model.cuda()
bidir_attention = bidir_attention.cuda()
ln_mlp = ln_mlp.cuda()
regressor = regressor.cuda()
torch.backends.cudnn.benchmark = True
base_model = nn.DataParallel(base_model)
bidir_attention = nn.DataParallel(bidir_attention)
ln_mlp = nn.DataParallel(ln_mlp)
regressor = nn.DataParallel(regressor)

base_model.apply(fix_bn)  # fix bn
base_model.eval()
bidir_attention.eval()
# ln_mlp.eval()
regressor.eval()
torch.set_grad_enabled(False)

# load data
train_trans, test_trans = get_video_trans()
# train_dataset = SevenPair_all_Dataset(class_idx_list=args.class_idx_list, score_range=100, subset='train',
#                                           data_root='/home/share/AQA_7/',
#                                           # data_root='../../dataset/Seven/',
#                                           transform=train_trans, frame_length=102, num_exemplar=1)
test_dataset = SevenPair_all_Dataset(class_idx_list=class_idx_list, score_range=5, subset='test',
                                         data_root='/mnt/gdata/AQA/AQA-7/',
                                         transform=test_trans, frame_length=102, num_exemplar=num_exemplars)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                  shuffle=False, num_workers=int(1),
                                                  pin_memory=True)

true_scores = []
pred_scores = []

for (data_1, data_2_list) in tqdm(test_dataloader):
    M = len(data_2_list)
    batch_size = data_1['final_score'].shape[0]
    pred_score_1_sum = torch.zeros((batch_size, 1)).cuda()
    # Data preparing for data_1
    video_1 = data_1['video'].float().cuda()  # N, C, T, H, W
    label_1 = data_1['final_score'].float().reshape(-1, 1).cuda()
    for data_2 in data_2_list:
        # Data preparing for data_2
        video_2 = data_2['video'].float().cuda()  # N, C, T, H, W
        label_2 = data_2['final_score'].float().reshape(-1, 1).cuda()

        # Forward
        # 1: pass backbone & attention
        feat_1, feat_2 = base_model(video_1, video_2)  # [B, 10, 1024]
        feature_2to1, feature_1to2, attn1, atttn2 = bidir_attention(feat_1, feat_2)  # [B, 10, 1024]
        # 2: concat features
        cat_feat_1 = torch.cat((feat_1, feature_2to1), 2)
        cat_feat_2 = torch.cat((feat_2, feature_1to2), 2)
        # 3: features fusion
        cat_feat_1 = ln_mlp(cat_feat_1)
        cat_feat_2 = ln_mlp(cat_feat_2)
        aggregated_feature_1 = cat_feat_1.mean(1)
        aggregated_feature_2 = cat_feat_2.mean(1)
        # 4: concat labels
        aggregated_feature_1 = torch.cat((aggregated_feature_1, label_2), 1)
        aggregated_feature_2 = torch.cat((aggregated_feature_2, label_1), 1)
        # 5: regress the scores
        pred_score_1 = regressor(aggregated_feature_1)
        pred_score_2 = regressor(aggregated_feature_2)

        # Summing up each score
        pred_score_1_sum += pred_score_1

    pred_score_1_avg = pred_score_1_sum / M

    # Updating score lists
    true_scores.extend(data_1['final_score'].numpy())
    pred_scores.extend([i.item() for i in pred_score_1_avg])

# analysis on results
pred_scores = np.array(pred_scores)
true_scores = np.array(true_scores)
rho, p = stats.spearmanr(pred_scores, true_scores)
L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
      true_scores.shape[0]
print('Correlation: %.4f, L2: %.4f, RL2: %.4f' % (rho, L2, RL2))
