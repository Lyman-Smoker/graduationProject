import torch.nn as nn
import torch
import os



class Bidir_Attention(nn.Module):
    def __init__(self, dim=1024, mask=False):
        super(Bidir_Attention, self).__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.scale = 1. / dim ** 0.5
        self.use_mask = mask

    def forward(self, feature1, feature2, topk=4):
        """
        :param feature1: [B, 10, 1024]
        :param feature2: [B, 10, 1024]
        :param topk:
        :return:
        """
        b, n, d = feature1.shape
        # qkv:[B, 10, 3, 1024]
        qkv_1 = self.qkv(feature1).reshape(b, n, 3, -1)
        qkv_2 = self.qkv(feature2).reshape(b, n, 3, -1)
        # q, k, v:[B, 10, 1024]
        q_1, k_1, v_1 = qkv_1.permute(2, 0, 1, 3)
        q_2, k_2, v_2 = qkv_2.permute(2, 0, 1, 3)
        # dot: [B, 10, 10]
        dot_1 = (q_1 @ k_2.transpose(-2, -1)) * self.scale
        dot_2 = (q_2 @ k_1.transpose(-2, -1)) * self.scale
        # attn: [B, 10, 10]
        attn_1 = dot_1.softmax(dim=-1)
        attn_2 = dot_2.softmax(dim=-1)
        if self.use_mask:
            # mask:
            _, attn_1_topk = torch.topk(attn_1, topk, sorted=True)
            _, attn_2_topk = torch.topk(attn_2, topk, sorted=True)
            # initialize mask
            mask_1 = torch.zeros(attn_1.shape).cuda()
            mask_2 = torch.zeros(attn_2.shape).cuda()
            for each_b in range(b):
                mask_1[each_b] = mask_1[each_b].scatter(1, attn_1_topk[each_b], 1)
                mask_2[each_b] = mask_2[each_b].scatter(1, attn_2_topk[each_b], 1)
            # Hadamard product
            attn_1 = attn_1.mul(mask_1) / torch.sum(attn_1, dim=2).unsqueeze(1).transpose(1,2)
            attn_2 = attn_2.mul(mask_2) / torch.sum(attn_2, dim=2).unsqueeze(1).transpose(1, 2)

        # feature_itoj: [B, 10, 1024]
        feature_2to1 = attn_1 @ v_2
        feature_1to2 = attn_2 @ v_1

        return feature_2to1, feature_1to2


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,8'
    ba = Bidir_Attention(mask=True, ln_mlp=True)
    feat1 = torch.randn(1, 10, 1024)
    feat2 = torch.randn(1, 10, 1024)
    feature_2to1, feature_1to2 = ba(feat1, feat2)
    print(feature_1to2.shape)