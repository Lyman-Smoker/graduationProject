import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from scipy import stats
import random
import argparse
import time
import copy
# dataset
from dataset import SevenPair_all_Dataset, get_video_trans, worker_init_fn
# model
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

mpl.use('Agg')

torch.manual_seed(0);
torch.cuda.manual_seed_all(0);
random.seed(0);
np.random.seed(0)
torch.backends.cudnn.deterministic = True


# fix BatchNorm
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def save_checkpoint(base_model, bidir_attention, ln_mlp, regressor, optimizer, epoch, rho_best, RL2_min, exp_name):
    torch.save({
        'base_model': base_model.state_dict(),
        'bidir_attention': bidir_attention.state_dict(),
        'regressor': regressor.state_dict(),
        'ln_mlp': ln_mlp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'rho': rho_best,
        'RL2': RL2_min,
    }, os.path.join('./ckpt', exp_name + '.pth'))


# resume training
def resume_train(base_model, bidir_attention, ln_mlp, regressor, optimizer):
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)
    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)
    bidir_attention_ckpt = {k.replace("module.", ""): v for k, v in state_dict['bidir_attention'].items()}
    bidir_attention.load_state_dict(bidir_attention_ckpt)
    ln_mlp_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ln_mlp'].items()}
    ln_mlp.load_state_dict(ln_mlp_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    # parameter
    start_epoch = state_dict['epoch']
    epoch_best = state_dict['epoch']
    rho_best = state_dict['rho']
    RL2_min = state_dict['RL2']

    return start_epoch, epoch_best, rho_best, RL2_min


def run_net():
    print('Runner starting ... ')

    # load model
    base_model = I3D_backbone(I3D_class=400)
    if not args.resume:
        base_model.load_pretrain(args.pretrained_i3d_weight)
    bidir_attention = Bidir_Attention(dim=args.feature_dim, mask=args.mask)
    ln_mlp = LN_MLP(dim=2*args.feature_dim, use_ln=args.ln, use_mlp=args.mlp)
    regressor = MLP(in_dim=2*args.feature_dim+1)

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        bidir_attention = bidir_attention.cuda()
        ln_mlp = ln_mlp.cuda()
        regressor = regressor.cuda()
        torch.backends.cudnn.benchmark = True

    # optimizer & scheduler
    optimizer = optim.Adam([
        {'params': base_model.parameters(), 'lr': args.base_lr * args.lr_factor},
        {'params': bidir_attention.parameters()},
        {'params': ln_mlp.parameters()},
        {'params': regressor.parameters()}
    ], lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = None

    # parameter setting
    start_epoch = 0
    global epoch_best, rho_best, L2_min, RL2_min
    epoch_best = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    # load checkpoint
    if args.resume:
        start_epoch, epoch_best, rho_best, RL2_min = resume_train(base_model, bidir_attention, ln_mlp, regressor, optimizer)
        print('resume ckpts @ %d epoch( rho = %.4f, RL2 = %.4f)' % (
            start_epoch, rho_best, RL2_min))

    # DP
    base_model = nn.DataParallel(base_model)
    bidir_attention = nn.DataParallel(bidir_attention)
    ln_mlp = nn.DataParallel(ln_mlp)
    regressor = nn.DataParallel(regressor)

    # loss
    mse = nn.MSELoss().cuda()

    # load data
    train_trans, test_trans = get_video_trans()
    train_dataset = SevenPair_all_Dataset(class_idx_list=args.class_idx_list, score_range=100, subset='train',
                                          # data_root='/home/share/AQA_7/',
                                          data_root='../../dataset/Seven/',
                                          transform=train_trans, frame_length=102, num_exemplar=1)
    test_dataset = SevenPair_all_Dataset(class_idx_list=args.class_idx_list, score_range=100, subset='test',
                                         data_root='../../dataset/Seven/',
                                         transform=test_trans, frame_length=102, num_exemplar=args.num_exemplars)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_train,
                                                   shuffle=True, num_workers=int(args.workers),
                                                   pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_test,
                                                  shuffle=False, num_workers=int(args.workers),
                                                  pin_memory=True)

    # epoch iterations
    steps = ['train', 'test']
    for epoch in range(start_epoch, args.max_epoch):
        print('EPOCH:', epoch)

        for step in steps:
            print(step + ' step:')
            true_scores = []
            pred_scores = []
            if args.fix_bn:
                base_model.apply(fix_bn)  # fix bn
            if step == 'train':
                base_model.train()
                bidir_attention.train()
                regressor.train()
                ln_mlp.train()
                torch.set_grad_enabled(True)
                data_loader = train_dataloader
            else:
                base_model.eval()
                bidir_attention.eval()
                regressor.eval()
                ln_mlp.eval()
                torch.set_grad_enabled(False)
                data_loader = test_dataloader
            for (data_1, data_2_list) in tqdm(data_loader):
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
                    feature_2to1, feature_1to2 = bidir_attention(feat_1, feat_2)  # [B, 10, 1024]
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

                    # Computing loss
                    loss_1 = mse(pred_score_1, label_1)
                    loss_2 = mse(pred_score_2, label_2)
                    loss = loss_1 + loss_2

                    # Optimization
                    if step == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

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
            print('[%s] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f' % (step, epoch, rho, L2, RL2))

            if args.record_results:
                # save checkpoint
                if step == 'test':
                    if rho > rho_best:
                        print('___________________find new best___________________')
                        rho_best = rho
                        save_checkpoint(base_model, bidir_attention, ln_mlp, regressor, optimizer, epoch, rho, RL2,
                                        args.exp_name + '_%.4f_%.4f@_%d' % (rho, RL2, epoch))
                    elif RL2 < RL2_min:
                        print('___________________find new best___________________')
                        RL2_min = RL2
                        save_checkpoint(base_model, bidir_attention, ln_mlp, regressor, optimizer, epoch, rho, RL2,
                                        args.exp_name + '_%.4f_%.4f@_%d' % (rho, RL2, epoch))
                # log
                writer.add_scalar(step + ' rho', rho, epoch)
                writer.add_scalar(step + ' L2', L2, epoch)
                writer.add_scalar(step + ' RL2', RL2, epoch)
                with open(args.exp_root + args.log_file, 'a') as f:
                    f.write(
                        '[%s] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f' % (step, epoch, rho, L2, RL2) + '\n')
                    f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test the model")
    parser.add_argument("--ckpt_path", type=str,
                        default='./ckpt/divinglast.pth',
                        help='path to the checkpoint model')
    parser.add_argument("--pretrained_i3d_weight", type=str,
                        default='../../pretrained_models/i3d_model_rgb.pth',
                        help='path to the checkpoint model')
    parser.add_argument("--class_idx_list", type=list,
                        default=[5],
                        help='path to the pretrained model')
    parser.add_argument("--feature_dim", type=int,
                        default=1024)
    parser.add_argument("--batch_size_test", type=int,
                        default=2,
                        help='batch size for testing')
    parser.add_argument("--batch_size_train", type=int,
                        default=2,
                        help='batch size for training')
    parser.add_argument("--workers", type=int,
                        default=1,
                        help='number of dataloader workers')
    parser.add_argument("--base_lr", type=float,
                        default=0.001,
                        help='base learning rate')
    parser.add_argument("--lr_factor", type=float,
                        default=0.1,
                        help='learning rate factor')
    parser.add_argument("--weight_decay", type=float,
                        default=0.001,
                        help='weight_decay')
    parser.add_argument("--resume", type=bool,
                        default=False,
                        help='load ckpt or not')
    parser.add_argument("--fix_bn", type=bool,
                        default=True,
                        help='fix bn or not')
    parser.add_argument("--max_epoch", type=int,
                        default=300)
    # 以下为每次实验必须仔细修改的选项
    parser.add_argument("--exp_name", type=str,
                        default='sync_diving_3m_ex10_mlp',
                        help='action_name + num_exemplar + (ln) + (mlp) + (mask)')
    # 同时跑两个实验 一定 一定 一定 要改下面这一项
    parser.add_argument("--log_file", type=str,
                        default='sync_diving_3m_ex10_mlp_log.txt')
    parser.add_argument("--exp_root", type=str,
                        default='./exp/')
    parser.add_argument("--record_results", type=bool,
                        default=True)
    # 消融
    parser.add_argument("--ln", type=bool,
                        default=False)
    parser.add_argument("--mlp", type=bool,
                        default=True)
    parser.add_argument("--num_exemplars", type=int,
                        default=10)
    parser.add_argument("--mask", type=bool,
                        default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    if args.record_results:
        if os.path.exists(args.exp_root + args.log_file):
            print('Remove former log file')
            os.remove(args.exp_root + args.log_file)
        else:
            print('There isn\'t a log file')
    writer = SummaryWriter('./exp/tensorboard/' + args.exp_name)
    run_net()
