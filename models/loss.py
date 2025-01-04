# models/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        # 计算相似度矩阵
        logits = logit_scale.exp() * image_features @ text_features.t()

        # 创建标签（对角矩阵）
        labels = torch.arange(len(logits)).to(logits.device)

        # 计算图像到文本和文本到图像的损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        # 总损失
        total_loss = (loss_i2t + loss_t2i) / 2
        return total_loss