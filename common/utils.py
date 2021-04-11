import torch
import numpy as np


def to_devices(tensors, device):
    """ helper function to send tensors to device """
    return (tensor.to(device) for tensor in tensors)


def loss_fn(y_pred, y_true, criterion):
    """ masked loss function """
    _mask = torch.logical_not(y_true == 0).float()
    _loss = criterion(y_pred, y_true)
    return (_loss * _mask).sum() / _mask.sum()


def accuracy_fn(y_pred, y_true):
    """ masked accuracy function """
    _mask = torch.logical_not(y_true == 0).float()
    _acc = (torch.argmax(y_pred, axis=-1) == y_true)
    return (_acc * _mask).sum() / _mask.sum()


def sample_direction(data, langs):
    """randomly sample a source and target language from
    n_langs possible combinations"""
    source, target = np.random.choice(len(langs), size=(2,), replace=False)
    return (data[source], data[target]), (langs[source], langs[target])


def mask_after_stop(input_tensor, stop_token):
    """mask all tokens after the stop token"""
    input_tensor = input_tensor.cpu()
    input_arr = input_tensor.numpy()
    match_indices = np.argwhere(input_arr == stop_token)
    batch_size, tensor_len = input_tensor.shape
    t_len_arr = np.vstack((np.arange(batch_size), np.ones(batch_size)*tensor_len)).T
    comb_arr = np.concatenate((match_indices, t_len_arr))
    _, i = np.unique(comb_arr[:, 0], return_index=True)
    end_indices = comb_arr[i][:, 1]

    mask = torch.zeros(input_tensor.shape[0], input_tensor.shape[1] + 1, dtype=int)
    mask[(torch.arange(input_tensor.shape[0]), end_indices)] = 1
    mask = 1 - mask.cumsum(dim=1)[:, :-1]
    output_tensor = input_tensor * mask

    return output_tensor


if __name__ == '__main__':
    a = torch.tensor([[1, 3, 4, 6, 2, 8, 8, 8], [4, 3, 2, 8, 8, 8, 8, 8], [1, 5, 6, 7, 4, 8, 9, 4]])
    print(mask_after_stop(a, stop_token=2))

