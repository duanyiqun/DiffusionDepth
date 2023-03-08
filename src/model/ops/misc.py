import numpy as np
import torch 
from torch import Tensor
import numba 


@numba.njit()
def pad_ones_np(array, dim=0):
    shape = list(array.shape)
    shape[dim] = 1
    return np.concatenate((array, np.ones(shape, array.dtype)), axis=dim)

def pad_ones_torch(tensor, dim=0):
    shape = list(tensor.shape) 
    shape[dim] = 1
    return torch.cat((tensor, tensor.new_ones(shape)), dim=dim)

def pad_ones(arr, dim=0):
    if isinstance(arr, Tensor):
        return pad_ones_torch(arr, dim)
    assert isinstance(arr, np.ndarray), type(arr)
    return pad_ones_np(arr, dim)


@numba.njit()
def pad_zeros_np(array, dim=0, n=1):
    shape = list(array.shape) 
    shape[dim] = n
    return np.concatenate((array, np.zeros(shape, array.dtype)), axis=dim)


def pad_zeros_torch(tensor, dim=0, n=1):
    shape = list(tensor.shape) 
    shape[dim] = n
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=dim)


def pad_zeros(arr, dim=0, n=1):
    if isinstance(arr, Tensor):
        return pad_zeros_torch(arr, dim, n)
    assert isinstance(arr, np.ndarray), type(arr)
    return pad_zeros_np(arr, dim, n)


# @numba.njit()
def pad_constants_np(array, value, dim, n):
    shape = list(array.shape)
    shape[dim] = n
    return np.concatenate((array, np.full(shape, value, dtype=array.dtype)), axis=dim)


def pad_constants_torch(tensor, value, dim, n): 
    shape = list(tensor.shape) 
    shape[dim] = n
    return torch.cat((tensor, tensor.full(shape, value)), dim=dim)


def pad_constants(arr, value, dim=0, n=1): 
    if isinstance(arr, Tensor): 
        return pad_constants_torch(arr, value, dim, n)
    assert isinstance(arr, np.ndarray), type(arr)
    return pad_constants_np(arr, value, dim, n)

