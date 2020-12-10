import torch
import torch.nn as nn
from DualTaskLoss import DualTaskLoss
import torch.nn.functional as F


class JointEdgeSegLoss(nn.Module):

    def __init__(self, classes=6, edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight
        self.nll_loss = nn.NLLLoss2d(ignore_index=255)

        self.dual_loss = DualTaskLoss().cuda()
        self.seg_loss = nn.CrossEntropyLoss().cuda()

    def bce2d(self, input, target):
        # c=1
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)


        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)


        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num


        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        loss = 0.0
        filler = torch.ones_like(target) * 255
        targets = torch.where(edge.max(1)[0] > 0.8, target, filler)
        for i in range(0, input.shape[0]):
            loss += self.nll_loss(F.log_softmax(input[i].unsqueeze(0)),
                                  targets[i].unsqueeze(0))
        return loss

    def forward(self, inputs, targets):
        segin, edgein = inputs
        segmask, edgemask = targets

        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)
        losses['edge_loss'] = self.edge_weight * self.bce2d(edgein, edgemask)
        losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein)
        losses['dual_loss'] = self.dual_weight * self.dual_loss(segin, segmask)

        return losses









