"""
Inference Loop for MNMT
"""

import numpy as np
import torch
import models.base_transformer as base_transformer
import utils.metrics as metrics
import utils.preprocess as preprocess


def test_step(x, y, model, criterion):
    # get masks and targets
    y_inp, y_tar = y[:, :-1], y[:, 1:]
    enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)

    # forward
    model.eval()
    y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)
    loss = criterion(y_pred.permute(0, 2, 1), y_tar)

    # metrics
    batch_loss = loss.item()
    batch_acc = (torch.argmax(y_pred.detach(), axis=-1) == y_tar).numpy().mean()
    batch_bleu = metrics.compute_bleu(y_tar, y_pred)

    return batch_loss, batch_acc, batch_bleu