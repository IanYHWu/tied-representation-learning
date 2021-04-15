import torch
import numpy as np
from itertools import combinations


def to_devices(tensors, device):
    """ helper function to send tensors to device """
    return (tensor.to(device) for tensor in tensors)


def loss_fn(y_pred, y_true, criterion):
    """ masked loss function """
    _mask = torch.logical_not(y_true == 0).float()
    _loss = criterion(y_pred, y_true)
    return (_loss * _mask).sum() / _mask.sum()


def auxiliary_loss_fn(x_enc, y_enc, criterion, x_mask=None, y_mask=None):
    """Computes loss according to criterion by masking and max pooling x_enc and y_enc."""

    # scale masks and reshape to (batch, seq_len, 1)
    x_mask = -9999.0 * x_mask.reshape(x_mask.size(0), -1, 1)
    y_mask = -9999.0 * y_mask.reshape(y_mask.size(0), -1, 1)

    # max pooling (batch, 1)
    x_enc = torch.max(x_enc + x_mask, dim=1)[0]
    y_enc = torch.max(y_enc + y_mask, dim=1)[0]

    return criterion(x_enc, y_enc)


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


def get_all_directions(data, langs):
    """unpacks all translation pairs from a batch of translations. Takes
    a dict of batches of tensors and returns a tensor of first dim size
    batch_size * len(langs) * (len(langs) - 1)."""

    pairs = list(combinations(langs, 2))
    pairs.extend([(y,x) for x,y in pairs])
    source, target = list(zip(*pairs))

    batch_size = data[0].shape[0]
    full_targets = []
    for t in target:
        full_targets.extend(batch_size * [t])

    x = torch.nn.utils.rnn.pad_sequence([data[langs.index(s)].t() for s in source], padding_value = 0)
    y = torch.nn.utils.rnn.pad_sequence([data[langs.index(t)].t() for t in target], padding_value = 0)

    x = x.flatten(start_dim = 1).t() # (batch * n_directions, max_len)
    y = y.flatten(start_dim = 1).t()

    return x, y, full_targets


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

