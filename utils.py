# utils.py: utility functions 
import os
import torch
import torch.nn.functional as F
import numpy as np


def pashamax(x : torch.Tensor, dim : int, eps : float = 1e-6):
    unnormed_logits = F.relu(x.exp()-1)
    return unnormed_logits / (unnormed_logits.sum(dim=dim, keepdim=True)+eps)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def sinusoidal_positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads#[np.newaxis, ...]

    return pos_encoding