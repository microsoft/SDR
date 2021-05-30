import os
import sys

import torch
import pickle
import numpy as np

from sklearn.preprocessing import normalize


def index_amp(lst, k):
    try:
        return lst.index(k) if k in lst else lst.index(k.replace("&", "&amp;"))
    except:
        return


def sim_matrix(a, b, eps=1e-8):
    """
    Similarity matrix
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

