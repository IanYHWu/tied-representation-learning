import torch
import numpy as np
from itertools import combinations


def to_devices(tensors, device, **kwargs):
    """ helper function to send tensors to device """
    return (tensor.to(device, **kwargs) for tensor in tensors)


def loss_fn(y_pred, y_true, criterion):
    """ masked loss function """
    _mask = torch.logical_not(y_true == 0).float()
    _loss = criterion(y_pred, y_true)
    return (_loss * _mask).sum() / _mask.sum()


def auxiliary_loss_fn(x_enc, y_enc, criterion, x_mask=None, y_mask=None):
    """Computes loss according to criterion by masking and max pooling x_enc and y_enc."""

    # scale masks and reshape to (batch, seq_len, 1)
    if x_mask is not None:
        x_mask = -9999.0 * x_mask.reshape(x_mask.size(0), -1, 1)
        x_enc += x_mask
    if y_mask is not None:
        y_mask = -9999.0 * y_mask.reshape(y_mask.size(0), -1, 1)
        y_enc += y_mask

    # max pooling (batch, emb_dim)
    x_enc = torch.max(x_enc, dim=1)[0]
    y_enc = torch.max(y_enc, dim=1)[0]

    return criterion(x_enc, y_enc)


def accuracy_fn(y_pred, y_true):
    """ masked accuracy function """
    _mask = torch.logical_not(y_true == 0).float()
    _acc = (torch.argmax(y_pred, axis=-1) == y_true)
    return (_acc * _mask).sum() / _mask.sum()


def sample_direction(data, langs, excluded=None):
    """randomly sample a source and target language from
    n_langs possible combinations"""
    source, target = np.random.choice(len(langs), size=(2,), replace=False)
    n_langs = len(langs)
    p = np.ones((n_langs,n_langs))
    np.fill_diagonal(p, 0.0)
    p = (p/p.sum()).flatten()
    for s, t in excluded:
        p[langs.index(s), langs.index(t)] = 0.0
    n = np.random.choice(n_langs**2, p=p)
    source, target = n // n_langs, n % n_langs
    return (data[source], data[target]), (langs[source], langs[target])


def get_direction(data, ind_source, ind_target):
    """extract a translation direction from the dataloader -
       deterministic version of sample_direction"""
    return data[ind_source], data[ind_target]


def get_pairs(inp_list, excluded=None):
    """get all ordered pairs of given list
    excluded : list of tuples of pairs to ignore."""
    pairs = list(combinations(inp_list, 2))
    pairs.extend([(y,x) for x,y in pairs])

    if excluded is not None:
        pairs = [(s, t) for s, t in pairs if (s, t) not in excluded]

    return pairs


def get_directions(data, langs, excluded=None):
    """unpacks all translation pairs from a batch of translations. Takes
    a dict of batches of tensors and returns a dict of tensors for each
    language direction.
    excluded : list of tuples of pairs to ignore.
    """

    source, target = list(zip(*get_pairs(langs, excluded=excluded)))

    batch_size = data[0].shape[0]
    out = {}
    for s, t in zip(source, target):
        target_lang = batch_size * [t]
        x = torch.nn.utils.rnn.pad_sequence(data[langs.index(s)].t(), padding_value = 0)
        y = torch.nn.utils.rnn.pad_sequence(data[langs.index(t)].t(), padding_value = 0)
        out[s+'-'+t] = (x, y, target_lang)

    return out


def get_all_directions(data, langs, excluded=None):
    """unpacks all translation pairs from a batch of translations. Takes
    a dict of batches of tensors and returns a tensor of first dim size
    batch_size * len(langs) * (len(langs) - 1).
    excluded : list of tuples of pairs to ignore."""

    source, target = list(zip(*get_pairs(langs, excluded=excluded)))

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
    mask = torch.roll((input_tensor == stop_token).int(), 1, dims=1)
    mask[:,0] = torch.zeros(mask.size(0)).to(mask.device)
    mask = 1 - torch.cumsum(mask, 1)
    return input_tensor * mask


def remove_after_stop(sentence):
    try:
        end_index = sentence.index('.')
    except ValueError:
        end_index = None
    if end_index is not None:
        if len(sentence) > end_index + 1:
            sentence = sentence[:end_index + 1]

    return sentence


if __name__ == "__main__":
    print(remove_after_stop(["."]))













