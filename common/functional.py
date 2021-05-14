import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
from models import base_transformer
from common.utils import to_devices, accuracy_fn, loss_fn, auxiliary_loss_fn


def train_step(x, y, model, criterion, aux_criterion, optimizer, scheduler, device, distributed=False):
    # get masks and targets
    y_inp, y_tar = y[:, :-1], y[:, 1:]
    enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)

    # mask for the target language encoded representation.
    enc_mask_aux = base_transformer.create_mask(y_inp)

    # devices
    x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask, enc_mask_aux = to_devices(
        (x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask, enc_mask_aux),
        device, non_blocking=distributed)

    # forward
    model.train()
    x_enc = model.encode(x, enc_mask)
    y_enc = model.encode(y_inp, enc_mask_aux)
    y_pred = model.final_layer(model.decode(y_inp, x_enc, look_ahead_mask, dec_mask)[0])
    loss = loss_fn(y_pred.permute(0, 2, 1), y_tar, criterion)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        loss_aux = auxiliary_loss_fn(x_enc, y_enc, aux_criterion, x_mask=enc_mask, y_mask=enc_mask_aux)

    # metrics
    batch_loss = loss.detach()
    batch_aux = loss_aux
    batch_acc = accuracy_fn(y_pred.detach(), y_tar)

    return batch_loss, batch_aux, batch_acc


def param_freeze(model, frozen_layers, unfreeze=False):
    """freeze parameters of encoder layers for any layer in frozen_layers."""
    for i, layer in enumerate(model.encoder.enc_layers):
        if i in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = unfreeze
    return model


def aux_train_step(x, y, model, criterion, aux_criterion, aux_strength, frozen_layers,
    optimizer, scheduler, device, distributed=False):
    """ Single training step using an auxiliary loss on the encoder outputs."""

    # get masks and targets
    y_inp, y_tar = y[:, :-1], y[:, 1:]
    enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)

    # mask for the target language encoded representation.
    enc_mask_aux = base_transformer.create_mask(y_inp)

    # devices
    x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask, enc_mask_aux = to_devices(
        (x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask, enc_mask_aux),
        device, non_blocking=distributed)

    model.train()
    optimizer.zero_grad()

    x_enc = model.encode(x, enc_mask)
    y_pred = model.final_layer(model.decode(y_inp, x_enc, look_ahead_mask, dec_mask)[0])
    y_enc = model.encode(y_inp, enc_mask_aux)

    # main loss.
    loss_main = loss_fn(y_pred.permute(0, 2, 1), y_tar, criterion)
    loss_main.backward(retain_graph=True)

    # aux loss
    model = param_freeze(model, frozen_layers)
    loss_aux = auxiliary_loss_fn(x_enc, y_enc, aux_criterion, x_mask=enc_mask, y_mask=enc_mask_aux)
    scaled_loss_aux = loss_aux * aux_strength
    scaled_loss_aux.backward()

    optimizer.step()
    scheduler.step()
    model = param_freeze(model, frozen_layers, unfreeze=True)

    # metrics
    batch_loss = loss_main.detach()
    batch_aux = loss_aux.detach()
    batch_acc = accuracy_fn(y_pred.detach(), y_tar)

    return batch_loss, batch_aux, batch_acc


def val_step(x, y, model, criterion, bleu, device, distributed=False):
    # get masks and targets
    y_inp, y_tar = y[:, :-1], y[:, 1:]
    enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)

    # devices
    x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask = to_devices(
        (x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask),
        device, non_blocking=distributed)

    # forward
    model.eval()
    with torch.no_grad():
        y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)
        loss = loss_fn(y_pred.permute(0, 2, 1), y_tar, criterion)

    # metrics
    batch_loss = loss.detach()
    batch_acc = accuracy_fn(y_pred.detach(), y_tar)

    bleu(torch.argmax(y_pred, axis=-1), y_tar)

    return batch_loss, batch_acc


def greedy_search(x, y, y_tar, model, enc_mask=None):
    """Inference loop taking the most probable token at each step."""
    x_enc = model.encoder(x, enc_mask)
    y_pred = []  # List to store predictions made by model.

    # loop to compute output of decoder without teacher forcing
    for t in range(y_tar.size(1)):
        with torch.no_grad():
            # Compute output of all translations up until sequence position t.
            output = model.final_layer(model.decode(y, x_enc, None, None)[0])
            # Take most recently computed time step.
            output = output[:, -1, :].squeeze()
            # Retrieve predicted token.
            max_token = torch.argmax(output, dim=-1).unsqueeze(-1)
            y = torch.cat([y, max_token], dim=1)
            y_pred.append(max_token)

    return torch.cat(y_pred, dim=1)


def single_beam_search(x, y, max_len, model, enc_mask=None, beam_length=2, alpha=0.0, beta=0.0):
    """
    x : (seq_len)
    y : ([]) tensor of start token
    max_len : legnth to decode to 
    """
    attn_block = 'decoder_layer' + str(model.num_layers) + '_block2'

    if enc_mask is not None:
        enc_mask = enc_mask.unsqueeze(0)

    x_enc = model.encode(x.unsqueeze(0), enc_mask) # (1, seq_len, d_model)
    x_enc = x_enc.repeat(beam_length, 1, 1)

    # first iteration
    y_pred, attn_weights = model.decode(y.reshape(1, 1), x_enc, None, None)
    y_pred = F.log_softmax(model.final_layer(y_pred), dim=-1)[:1, -1, :] # (1, vocab)

    # compute beam score by s = logP / lp + cp
    # cp
    attn_weights = attn_weights[attn_block].mean(1).sum(1) # (1, x_len)
    cp = beta * torch.log(torch.clamp(attn_weights, min=1.0)).sum(-1) # (1,)

    # lp
    lp = 1.0

    # s
    s = (y_pred / lp + cp.unsqueeze(-1)).reshape(-1) # (vocab)
    
    # trim beams
    _, new_token = torch.topk(s, beam_length) # (beam,)
    log_p = y_pred[0][new_token]
    y = torch.cat([y.repeat(beam_length, 1), new_token.unsqueeze(-1)], dim=-1) # (beam, 2)

    for t in range(1, max_len):
        with torch.no_grad():

            # expand beams
            y_dec, attn_weights = model.decode(y, x_enc, None, None)
            y_pred = F.log_softmax(model.final_layer(y_dec), dim=-1)[:, -1, :] # (beam, vocab)
            new_log_p = (log_p.unsqueeze(-1) + y_pred)

            # compute score
            attn_weights = attn_weights[attn_block].mean(1).sum(1) # (beam, x_len)
            cp = beta * torch.log(torch.clamp(attn_weights, min=1.0)).sum(-1) # (beam,)
            lp = ((y.size(1) + 5) ** alpha) / (6 ** alpha) # (beam, vocab)
            s = (new_log_p / lp + cp.unsqueeze(-1)).reshape(-1) # (beam * vocab)

            # trim beams
            _, beam_idxs = torch.topk(s, beam_length) # (beam,)
            beam_id, new_token = beam_idxs // y_pred.size(-1), beam_idxs % y_pred.size(-1)
            log_p = new_log_p[beam_id, new_token]

            # update input
            y = torch.cat([y[beam_id], new_token.unsqueeze(-1)], dim=-1)

    best_beam = log_p.argmax()
    y = y[best_beam]

    return y[1:]
    

def beam_search(x, y, y_tar, model, enc_mask=None, beam_length=2, alpha=0.0, beta=0.0):
    preds = []
    for i in range(x.size(0)):
        enc_mask_i = enc_mask[i] if enc_mask is not None else None
        preds.append(single_beam_search(
            x[i], y[i], y_tar.size(1), model,
            enc_mask=enc_mask_i, beam_length=beam_length,
            alpha=alpha, beta=beta)
        )
    return torch.stack(preds, dim=0)


def inference_step(x, y, model, logger, tokenizer, device, bleu=None,
                   teacher_forcing=False, pivot_mode=False, beam_length=1,
                   alpha=0.0, beta=0.0):
    """
    inference step.
    x: source language
    y: target language
    """
    if teacher_forcing:
        y_inp, y_tar = y[:, :-1], y[:, 1:]
        enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)

        # devices
        x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask = to_devices(
            (x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask),
            device)

        # inference
        model.eval()
        with torch.no_grad():
            y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)

        if not pivot_mode:
            batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()
            bleu(torch.argmax(y_pred, axis=-1), y_tar)
            logger.log_examples(x, y_tar, torch.argmax(y_pred, axis=-1), tokenizer)
            return batch_acc
        else:
            return torch.argmax(y_pred, axis=-1)

    else:
        # Retrieve the start of sequence token and the target translation
        y, y_tar = y[:, 0].unsqueeze(-1), y[:, 1:]
        enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_tar)

        # devices
        x, y, y_tar, enc_mask = to_devices((x, y, y_tar, enc_mask), device)

        # inference
        model.eval()
        if beam_length == 1:
            y_pred = greedy_search(x, y, y_tar, model, enc_mask=enc_mask)
        else:
            y_pred = beam_search(x, y, y_tar, model, enc_mask=enc_mask, beam_length=beam_length,
                alpha=alpha, beta=beta)

        if not pivot_mode:
            batch_acc = 0
            if bleu is not None:
                bleu(y_pred, y_tar)
            logger.log_examples(x, y_tar, y_pred, tokenizer)
            return batch_acc
        else:
            return y_pred

