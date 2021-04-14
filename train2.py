"""
Training Loop for MNMT
"""

import time
from copy import deepcopy

import numpy as np
import torch
from einops import rearrange

import models.base_transformer as base_transformer
import models.initialiser as initialiser
from common import data_logger as logging
from common import preprocess
from common.train_arguments import train_parser
from hyperparams.loader import Loader
from hyperparams.schedule import WarmupDecay


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
    """ randomly sample a source and target language from
    n_langs possible combinations. """
    source, target = np.random.choice(len(langs), size=(2,), replace=False)
    return (data[source], data[target]), (langs[source], langs[target])


def param_freeze(model, frozen_layers):
    # create deepcopy of main model.
    model2 = deepcopy(model)

    # here we freeze specific parameters in the auxiliary model.
    for i, param in enumerate(model2.encoder.enc_layers.parameters()):
        if i in frozen_layers:
            param.requires_grad = False
    return model2


def train_step(x, y, model, model2, criterion, aux_criterion, target, optimizer1, optimizer2, scheduler, device):
    # get masks and targets
    y_inp, y_tar = y[:, :-1], y[:, 1:]
    enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)
    # # need to create a mask for the target language encoded representation.
    enc_mask_aux = base_transformer.create_mask(y_inp)

    # devices
    x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask, enc_mask_aux = to_devices(
        (x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask, enc_mask_aux),
        device)

    # forward
    model.train()
    model2.train()

    y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)

    # flatten sequences to bag of words representation.
    x_enc = model2.encoder(x, enc_mask)
    x_enc = rearrange(x_enc, "b h n -> b (h n)")
    y_enc = model2.encoder(y_tar, enc_mask_aux)
    y_enc = rearrange(y_enc, "b h n -> b (h n)")

    # main loss.
    loss1 = loss_fn(y_pred.permute(0, 2, 1), y_tar, criterion)
    # auxiliary loss.
    if x_enc.size(1) > y_enc.size(1):
        x_enc = x_enc[:, :y_enc.size(1)]
    elif y_enc.size(1) > x_enc.size(1):
        y_enc = y_enc[:, :x_enc.size(1)]

    loss2 = aux_criterion(x_enc, y_enc, target)

    loss = loss1 + loss2

    # backward
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss.backward()
    optimizer1.step()
    optimizer2.step()
    scheduler.step()

    # Average out parameter updates for main model.
    for state in model.state_dict():
        model.state_dict()[state] = (model.state_dict()[state] + model2.state_dict()[state]) / 2

    # load main model parameters into the state dictionary of auxiliary model.
    model2.load_state_dict(model.state_dict())

    # metrics
    batch_loss = loss.cpu().item()
    batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()

    return batch_loss, batch_acc


def val_step(x, y, model, criterion, device):
    # get masks and targets
    y_inp, y_tar = y[:, :-1], y[:, 1:]
    enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)

    # devices
    x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask = to_devices(
        (x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask),
        device)

    # forward
    model.eval()
    with torch.no_grad():
        y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)
        loss = loss_fn(y_pred.permute(0, 2, 1), y_tar, criterion)

    # metrics
    batch_loss = loss.item()
    batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()

    return batch_loss, batch_acc


def train(device, params, train_dataloader, val_dataloader=None, tokenizer=None, frozen_layers=[]):
    """Training Loop"""

    multi = len(params.langs) > 2
    if multi:
        assert tokenizer is not None
        add_targets = preprocess.AddTargetTokens(params.langs, tokenizer)

    new_root_path = params.location
    new_name = params.name
    if params.checkpoint:
        params = logging.load_params(new_root_path + '/' + new_name)
        params.location = new_root_path
        params.name = new_name
        logger = logging.TrainLogger(params)
        logger.make_dirs()
    else:
        logger = logging.TrainLogger(params)
        logger.make_dirs()
    logger.save_params()

    model = initialiser.initialise_model(params, device)
    model2 = param_freeze(model, frozen_layers)

    optimizer1 = torch.optim.Adam(model.parameters())
    optimizer2 = torch.optim.Adam(model2.parameters())
    scheduler = WarmupDecay(optimizer1, params.warmup_steps, params.d_model, lr_scale=params.lr_scale)
    criterion1 = torch.nn.CrossEntropyLoss(reduction='none')
    criterion2 = torch.nn.CosineEmbeddingLoss(reduction='mean')

    epoch = 0
    if params.checkpoint:
        model, optimizer1, epoch = logging.load_checkpoint(logger.checkpoint_path, model, optimizer1)

    batch_losses, batch_accs = [], []
    epoch_losses, epoch_accs = [], []
    val_epoch_losses, val_epoch_accs, val_epoch_bleus = [], [], []
    target = torch.tensor([1])

    while epoch < params.epochs:
        start_ = time.time()

        # train
        epoch_loss = 0.0
        epoch_acc = 0.0
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        for i, data in enumerate(train_dataloader):

            if multi:
                # sample a tranlsation direction and add target tokens
                (x, y), (x_lang, y_lang) = sample_direction(data, params.langs)
                x = add_targets(x, y_lang)
            else:
                x, y = data

            batch_loss, batch_acc = train_step(x, y, model, model2, criterion1, criterion2, target, optimizer1,
                                               optimizer2, scheduler,
                                               device)

            batch_losses.append(batch_loss)
            batch_accs.append(batch_acc)

            epoch_loss += (batch_loss - epoch_loss) / (i + 1)
            epoch_acc += (batch_acc - epoch_acc) / (i + 1)

            if i % 50 == 0:
                print('Batch {} Loss {:.4f} Accuracy {:.4f} in {:.4f} s per batch'.format(
                    i, epoch_loss, epoch_acc, (time.time() - start_) / (i + 1)))

        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)

        # val
        if val_dataloader is not None:
            for i, data in enumerate(val_dataloader):
                if multi:
                    # sample a tranlsation direction and add target tokens
                    (x, y), (x_lang, y_lang) = sample_direction(data, params.langs)
                    x = add_targets(x, y_lang)
                else:
                    x, y = data

                batch_loss, batch_acc = val_step(x, y, model, criterion1, device)
                val_epoch_loss += (batch_loss - val_epoch_loss) / (i + 1)
                val_epoch_acc += (batch_acc - val_epoch_acc) / (i + 1)

            val_epoch_losses.append(val_epoch_loss)
            val_epoch_accs.append(val_epoch_acc)

            print('Epoch {} Loss {:.4f} Accuracy {:.4f} Val Loss {:.4f} Val Accuracy {:.4f}'
                  ' in {:.4f} secs \n'.format(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc,
                                              time.time() - start_))
        else:
            print('Epoch {} Loss {:.4f} Accuracy {:.4f} in {:.4f} secs \n'.format(
                epoch, epoch_loss, epoch_acc, time.time() - start_))

        epoch += 1

        logger.save_model(epoch, model, optimizer)
        logger.log_results([epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc])

    return epoch_losses, epoch_accs, val_epoch_losses, val_epoch_accs


def main(params):
    """ Loads the dataset and trains the model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(params.langs) == 2:
        # bilingual translation

        train_dataloader, val_dataloader, test_dataloader, _ = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset, multi=False)

        train(device, params, train_dataloader, val_dataloader=val_dataloader,frozen_layers = params.frozen_layers)
    else:
        # multilingual translation

        train_dataloader, val_dataloader, test_dataloader, tokenizer = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset, multi=True)

        train(device, params, train_dataloader, val_dataloader=val_dataloader, tokenizer=tokenizer, frozen_layers = params.frozen_layers)


if __name__ == "__main__":
    args = train_parser.parse_args()

    # Loader can also take in any dictionary of parameters
    params = Loader(args, check_custom=True)
    main(params)
