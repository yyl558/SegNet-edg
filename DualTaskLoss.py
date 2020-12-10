import torch
import torch.nn as nn
import torch.nn.functional as F
from my_functionals import compute_grad_mag

def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    assert logits.dim() == 3
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps)
    y = (logits.cuda()) + (gumbel_noise.cuda())
    return F.softmax(y / tau, 1)


def _one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """

    y = torch.eye(num_classes).cuda()
    return y[labels].permute(0,3,1,2)

def _sample_gumbel(shape, eps=1e-10):
    """
    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).cuda()
    return - torch.log(eps - torch.log(U + eps))


class DualTaskLoss(nn.Module):
    def __init__(self, cuda=False):
        super(DualTaskLoss, self).__init__()
        self._cuda = cuda
        return

    def forward(self, input_logits, gts):
        """
        :param input_logits: NxCxHxW segin
        :param gt_semantic_masks: NxCxHxW segmask
        :return: final loss
        """
        N, C, H, W = input_logits.shape
        th = 1e-8  # 1e-10
        eps = 1e-10

        gt_semantic_masks = gts.detach()
        gt_semantic_masks = gt_semantic_masks.long().cuda()
        y = torch.eye(6).cuda()
        gt_semantic_masks = y[gt_semantic_masks].permute(0,3,1,2)
        #gt_semantic_masks = _one_hot_embedding(gt_semantic_masks, 6).detach()


        g = _gumbel_softmax_sample(input_logits.view(N, C, -1), tau=0.5)

        g = g.reshape((N, C, H, W))
        g = compute_grad_mag(g, cuda=self._cuda)

        g_hat = compute_grad_mag(gt_semantic_masks, cuda=self._cuda)


        g = g.view(N, -1)

        g_hat = g_hat.reshape(N, -1)

        loss_ewise = F.l1_loss(g, g_hat, reduction='none', reduce=False)

        p_plus_g_mask = (g >= th).detach().float()
        loss_p_plus_g = torch.sum(loss_ewise * p_plus_g_mask) / (torch.sum(p_plus_g_mask) + eps)

        p_plus_g_hat_mask = (g_hat >= th).detach().float()
        loss_p_plus_g_hat = torch.sum(loss_ewise * p_plus_g_hat_mask) / (torch.sum(p_plus_g_hat_mask) + eps)

        total_loss = 0.5 * loss_p_plus_g + 0.5 * loss_p_plus_g_hat

        return total_loss
