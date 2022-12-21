import torch
import torch.nn as nn

import numpy as np

from ..attack import Attack

class HopSkipJump(Attack):
    r"""
    "HopSkipJumpAttack" in the paper 'HopSkipJumpAttack: A Query-Efficient Decision-Based Attack'
    [https://arxiv.org/abs/1904.02144]
    [https://github.com/Jianbo-Lab/HSJA]
    Distance Measure : L2, Linf
    Arguments:
        model (nn.Module): model to attack.
        clip_max (float): upper bound of the image.
        clip_min (float): lower bound of the image.
        norm (string): Lp-norm to minimize. ['L2', 'Linf'] (Default: 'L2')
        num_iterations (int): number of iterations. (Default: 20)


        n_queries (int): max number of queries (each restart). (Default: 5000)
        n_restarts (int): number of random restarts. (Default: 1)
        p_init (float): parameter to control size of squares. (Default: 0.8)
        loss (str): loss function optimized ['margin', 'ce'] (Default: 'margin')
        resc_schedule (bool): adapt schedule of p to n_queries (Default: True)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)
        targeted (bool): targeted. (Default: False)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.Square(model, model, norm='Linf', eps=8/255, n_queries=5000, n_restarts=1, eps=None, p_init=.8, seed=0, verbose=False, targeted=False, loss='margin', resc_schedule=True)
        >>> adv_images = attack(images, labels)
    """
