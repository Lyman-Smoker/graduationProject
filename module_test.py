import torch
# visualization
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def diversity_loss(attention):
    attention_t = torch.transpose(attention, 1, 2)
    num_features = attention.shape[1]
    # res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).cuda()
    res = (attention @ attention_t)
    dis = torch.norm(attention, 2, 2)
    dis = dis.unsqueeze(-1)
    dis_mask = dis @ dis.transpose(1, 2)
    res = res / dis_mask
    res = res - torch.eye(num_features)

    sns.heatmap(
        pd.DataFrame(np.round(res[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
    plt.savefig('./res.png')
    # plt.show()
    plt.close() 

    res = res.view(-1, num_features*num_features)
    return torch.norm(res, p=2, dim=1).sum() / attention.size(0)

a = torch.randn(1,10,10)
b = torch.transpose(a, 1, 2)
num_features = a.shape[1]
# res = torch.matmul(attention_t.view(-1, args.num_filters, num_features), attention.view(-1, num_features, args.num_filters)) - torch.eye(args.num_filters).cuda()
a_2 = a**2
ab = a @ b
ab_soft = (a @ b).softmax(dim=-1)
ab_I = (a @ b).softmax(dim=-1) - torch.eye(num_features)

dis = torch.norm(a, 2, 2)
dis = dis.unsqueeze(-1)
dis_mask = dis @ dis.transpose(1, 2)
ab_cos = ab / dis_mask
ab_cos_I = ab_cos - torch.eye(10)

_ = diversity_loss(a)

sns.heatmap(
    pd.DataFrame(np.round(ab[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
plt.savefig('./ab.png')
# plt.show()
plt.close()

sns.heatmap(
    pd.DataFrame(np.round(dis_mask[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
plt.savefig('./dis_mask.png')
# plt.show()
plt.close()

sns.heatmap(
    pd.DataFrame(np.round(ab_cos_I[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
plt.savefig('./ab_cos-I.png')
# plt.show()
plt.close()

sns.heatmap(
    pd.DataFrame(np.round(a[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
plt.savefig('./a.png')
# plt.show()
plt.close()

sns.heatmap(
    pd.DataFrame(np.round(a_2[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
plt.savefig('./a_2.png')
# plt.show()
plt.close()

sns.heatmap(
    pd.DataFrame(np.round(b[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
plt.savefig('./b.png')
# plt.show()
plt.close()

sns.heatmap(
    pd.DataFrame(np.round(ab_soft[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
plt.savefig('./AB_soft.png')
# plt.show()
plt.close()

sns.heatmap(
    pd.DataFrame(np.round(ab_I[0].detach().cpu().numpy(), 4)), annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="Blues", annot_kws={'size': 6})
plt.savefig('./AB-I.png')
# plt.show()
plt.close()