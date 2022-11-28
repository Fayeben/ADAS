from ctypes import c_ulong
from importlib.abc import ResourceLoader
from xml import dom
import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVDiscriminator_Conv_2(nn.Module):
    def __init__(self):
        super().__init__()
        # c_in = model_cfg['FEATURE_DIM']
        # c_out = model_cfg['FEATURE_DIM'] // 4
        c_in = 16
        c_in_2d = 64
        c_out = 64 // 4
        self.c_out = c_out
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.mlp1 = nn.Linear(c_in, c_out)
        self.mlp2 = nn.Linear(c_in_2d, c_out)
        self.mlp = nn.Linear(c_out, 1)

    def forward(self, x, domm='2d'):
        # x = batch_dict['spatial_features_2d']
        # score = batch_dict['bev_score']
        # bev_map = batch_dict['bev_map']
        # bev_map = F.sigmoid(bev_map).clamp_(1e-6)
        # entropy_map = -bev_map * torch.log2(bev_map) - (1 - bev_map) * torch.log2(1 - bev_map)
        # entropy_map = entropy_map.max(dim=1)[0].view(-1, 1, *entropy_map.shape[2:])
        # score = F.sigmoid(score)
        # score = (torch.cat([score, entropy_map], dim=1).mean(dim=1).view(-1, 1, *score.shape[2:])) / 2
        # x = x * (1 + score)
        # x = self.block(x)
        # x = self.gap(x).view(-1, self.c_out)
        if domm=='2d':
            feats = self.mlp2(x["feats"].detach())
        elif domm=='3d':
            feats = self.mlp1(x["feats"].detach())
        feats = self.mlp(feats)
        x["domainness"] = feats
        # feature_res = get_discriminator_loss(domainness)
        return x

    def get_discriminator_loss(self, x, source=True, loss='bce'):
        domainness = x["domainness"]
        if source:
            if loss == 'bce':
                discri_loss = bce_loss(domainness, 0)
            else:
                discri_loss = ls_loss(domainness, 0)
        else:
            if loss == 'bce':
                discri_loss = bce_loss(domainness, 1)
            else:
                discri_loss = ls_loss(domainness, 1)
        return discri_loss

    def domainness_evaluate(self, batch_dict, source=False):
        domainness = batch_dict['domainness'].cuda()
        # domainness_value = 1 / (math.sqrt(2*3.14) * self.model_cfg.SIGMA) * torch.exp(-(domainness - self.model_cfg.MU).pow(2) / 2 * (self.model_cfg.SIGMA ** 2))
        # batch_dict['domainness_evaluate'] = domainness_value
        batch_dict['domainness_evaluate'] = domainness
        batch_dict['domainness_evaluate_sigmoid'] = F.sigmoid(domainness)
        return batch_dict


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def ls_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.MSELoss()(y_pred, y_truth_tensor)