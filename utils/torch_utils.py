import numpy as np
import torch
import numbers

def mean_non_pad_value(tensor, axis, pad_value=0):
    mask = tensor != pad_value
    tensor[~mask] = 0
    tensor_mean = (tensor * mask).sum(dim=axis) / (mask.sum(dim=axis))

    ignore_mask = (mask.sum(dim=axis)) == 0
    tensor_mean[ignore_mask] = pad_value
    return tensor_mean


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, numbers.Number):
        return np.array(tensor)
    else:
        raise NotImplementedError
