"""
Training Loop for MNMT
"""
SEED = 11

import torch
import torch.nn as nn 
import torch.multiprocessing as mp 
import torch.distributed as dist 

import numpy as np 
import os, sys, time
import wandb
from tokenizers import Tokenizer

import models.base_transformer as base_transformer
import models.initialiser as initialiser
from common import preprocess
from common.train_arguments import train_parser
from common import data_logger as logging
from hyperparams.loader import Loader
from hyperparams.schedule import WarmupDecay
from common.metrics import BLEU
from common.utils import to_devices, accuracy_fn, loss_fn, auxiliary_loss_fn, sample_direction, get_direction

def seed_all(SEED):
    """ Set the seed for all devices. """
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)


def train_step(x, y, model, criterion, optimizer, scheduler, device, distributed=False):
    # get masks and targets
    y_inp, y_tar = y[:, :-1], y[:, 1:]
    enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)

    # devices
    x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask = to_devices(
        (x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask),
        device, non_blocking=distributed)

    # forward
    model.train()
    y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)
    loss = loss_fn(y_pred.permute(0, 2, 1), y_tar, criterion)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # metrics
    batch_loss = loss.detach()
    batch_acc = accuracy_fn(y_pred.detach(), y_tar)

    return batch_loss, batch_acc


def param_freeze(model, frozen_layers, unfreeze=False):
    """freeze parameters of encoder layers for any layer in frozen_layers."""
    for i, layer in enumerate(model.encoder.enc_layers):
        if i in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = unfreeze
    return model


def aux_train_step(x, y, model, criterion, aux_criterion, frozen_layers,
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

    x_enc = model.encoder(x, enc_mask)
    y_pred = model.final_layer(model.decoder(y_inp, x_enc, look_ahead_mask, dec_mask)[0])
    y_enc = model.encoder(y_inp, enc_mask_aux)

    # main loss.
    loss_main = loss_fn(y_pred.permute(0, 2, 1), y_tar, criterion)
    loss_main.backward(retain_graph=True)

    # aux loss
    model = param_freeze(model, frozen_layers)
    loss_aux = auxiliary_loss_fn(x_enc, y_enc, aux_criterion, x_mask=enc_mask, y_mask=enc_mask_aux)
    loss_aux.backward()

    optimizer.step()
    scheduler.step()
    model = param_freeze(model, frozen_layers, unfreeze=True)

    # metrics
    loss = loss_main + loss_aux
    batch_loss = loss.detach()
    batch_acc = accuracy_fn(y_pred.detach(), y_tar)

    return batch_loss, batch_acc


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


def setup(params):
    new_root_path = params.location
    new_name = params.name
    if params.checkpoint:
        add_epochs = params.add_epochs
        params = logging.load_params(new_root_path + '/' + new_name)
        params.location = new_root_path
        params.name = new_name
        params.epochs += add_epochs
        logger = logging.TrainLogger(params)
        logger.make_dirs()
    else:
        logger = logging.TrainLogger(params)
        logger.make_dirs()
    logger.save_params()
    return logger


def train(rank, device, logger, params, train_dataloader, val_dataloader=None, tokenizer=None,
    verbose=50, pivot=False, pivot_pair_ind=(0, 1)):
    """Training Loop
    For training a bilingual model as part of a pivot:
        use pivot=True, and multilingual dataloaders.
        specify the pivot pair in pivot_pair_ind
    """

    multi = False
    if len(params.langs) > 2 and not pivot:
        assert tokenizer is not None
        multi = True
        add_targets = preprocess.AddTargetTokens(params.langs, tokenizer)

    model = initialiser.initialise_model(params, device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = WarmupDecay(optimizer, params.warmup_steps, params.d_model, lr_scale=params.lr_scale)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    if params.auxiliary:
        _aux_criterion = torch.nn.CosineEmbeddingLoss(reduction='mean')
        _target = torch.tensor(1.0).to(device)
        aux_criterion = lambda x, y: params.aux_strength * _aux_criterion(x, y, _target)
    
    epoch = 0
    if params.checkpoint:
        model, optimizer, epoch, scheduler = logging.load_checkpoint(logger.checkpoint_path, device, model,
            optimizer=optimizer, scheduler=scheduler)
    
    if params.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index],
            find_unused_parameters=True)

    if rank == 0:
        if params.wandb:
            wandb.watch(model)
        batch_losses, batch_accs = [], []
        epoch_losses, epoch_accs = [], []
        val_epoch_losses, val_epoch_accs, val_epoch_bleus = [], [], []

    while epoch < params.epochs:
        start_ = time.time()

        # train
        epoch_loss = 0.0
        epoch_acc = 0.0
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        val_bleu = 0.0
        for i, data in enumerate(train_dataloader):

            if multi:
                # sample a tranlsation direction and add target tokens
                (x, y), (x_lang, y_lang) = sample_direction(data, params.langs)
                x = add_targets(x, y_lang)
            elif pivot:
                x, y = get_direction(data, pivot_pair_ind[0], pivot_pair_ind[1])
            else:
                x, y = data

            if params.auxiliary:
                batch_loss, batch_acc = aux_train_step(x, y, model, criterion, aux_criterion,
                    params.frozen_layers, optimizer, scheduler, device, distributed=params.distributed)
            else:
                batch_loss, batch_acc = train_step(x, y, model, criterion, optimizer, scheduler,
                    device, distributed=params.distributed)

            if rank == 0:
                batch_loss = batch_loss.item()
                batch_acc = batch_acc.item()
                batch_losses.append(batch_loss)
                batch_accs.append(batch_acc)
                epoch_loss += (batch_loss - epoch_loss) / (i + 1)
                epoch_acc += (batch_acc - epoch_acc) / (i + 1)

                if verbose is not None:
                    if i % verbose == 0:
                        print('Batch {} Loss {:.4f} Accuracy {:.4f} in {:.4f} s per batch'.format(
                            i, epoch_loss, epoch_acc, (time.time() - start_) / (i + 1)))
                if params.wandb:
                    wandb.log({'loss': epoch_loss, 'accuracy': epoch_acc})

        if rank == 0:
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)

        # val only on rank 0
        if rank == 0:
            if val_dataloader is not None:
                bleu = BLEU()
                bleu.set_excluded_indices([0, 2])
                for i, data in enumerate(val_dataloader):
                    if multi:
                        # sample a tranlsation direction and add target tokens
                        (x, y), (x_lang, y_lang) = sample_direction(data, params.langs)
                        x = add_targets(x, y_lang)
                    elif pivot:
                        x, y = get_direction(data, pivot_pair_ind[0], pivot_pair_ind[1])
                    else:
                        x, y = data

                    batch_loss, batch_acc = val_step(x, y, model, criterion, bleu, device,
                        distributed=params.distributed)
                    val_epoch_loss += (batch_loss - val_epoch_loss) / (i + 1)
                    val_epoch_acc += (batch_acc - val_epoch_acc) / (i + 1)

                val_epoch_losses.append(val_epoch_loss)
                val_epoch_accs.append(val_epoch_acc)
                val_bleu = bleu.get_metric()

                if verbose is not None:
                    print('Epoch {} Loss {:.4f} Accuracy {:.4f} Val Loss {:.4f} Val Accuracy {:.4f} Val Bleu {:.4f}'
                          ' in {:.4f} secs \n'.format(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, val_bleu,
                                                      time.time() - start_))
                if params.wandb:
                    wandb.log({'loss': epoch_loss, 'accuracy': epoch_acc, 'val_loss': val_epoch_loss,
                               'val_accuracy': val_epoch_acc, 'val_bleu': val_bleu})
            else:
                if verbose is not None:
                    print('Epoch {} Loss {:.4f} Accuracy {:.4f} in {:.4f} secs \n'.format(
                        epoch, epoch_loss, epoch_acc, time.time() - start_))
                if params.wandb:
                    wandb.log({'loss': epoch_loss, 'accuracy': epoch_acc, 'val_loss': val_epoch_loss,
                               'val_accuracy': val_epoch_acc, 'val_bleu': val_bleu})

        epoch += 1

        if rank == 0:
            logger.save_model(epoch, model, optimizer)
            logger.log_results([epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, val_bleu])

    return epoch_losses, epoch_accs, val_epoch_losses, val_epoch_accs


def main(gpu, params):
    """ Loads the dataset and trains the model."""
    rank = params.nr * params.gpus + gpu
    if params.distributed:
        dist.init_process_group(backend='nccl', init_method='env://',
            world_size=params.world_size, rank=rank)
    seed_all(SEED)

    # get gpu device
    device = torch.device(gpu)

    # only wandb on main process
    if rank == 0 and params.wandb:
        wandb.init(project='mnmt', entity='nlp-mnmt-project',
                   config={k: v for k, v in params.__dict__.items() if isinstance(v, (float, int, str))})
        config = wandb.config
    logger = setup(params)

    # load data and train for required experiment
    if len(params.langs) == 2 and not params.pivot:
        # bilingual translation

        # load tokenizers if continuing
        if params.checkpoint:
            tokenizers = []
            for lang in params.langs:
                tokenizers.append(Tokenizer.from_file(logger.root_path + '/' + lang + '_tokenizer.json'))
        else:
            tokenizers = None

        train_dataloader, val_dataloader, test_dataloader, _ = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset, multi=False, path=logger.root_path,
            tokenizer=tokenizers, distributed=params.distributed, world_size=params.world_size, rank=rank)

        train(rank, device, logger, params, train_dataloader, val_dataloader=val_dataloader, verbose=params.verbose)

    elif len(params.langs) > 2 and not params.pivot:
        # multilingual translation

        # load tokenizers if continuing
        if params.checkpoint:
            tokenizer = Tokenizer.from_file(logger.root_path + '/multi_tokenizer.json')
        else:
            tokenizer = None

        train_dataloader, val_dataloader, test_dataloader, tokenizer = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset, multi=True, path=logger.root_path,
            tokenizer=tokenizer, distributed=params.distributed, world_size=params.world_size, rank=rank)

        train(rank, device, logger, params, train_dataloader, val_dataloader=val_dataloader, tokenizer=tokenizer,
              verbose=params.verbose)

    elif len(params.langs) > 2 and params.pivot:
        # pivot translation

        if params.pivot_tokenizer_path:
            tokenizer = Tokenizer.from_file(params.pivot_tokenizer_path)
        else:
            tokenizer = None

        train_dataloader, val_dataloader, test_dataloader, tokenizer = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset, multi=True, path=logger.root_path,
            distributed=params.distributed, world_size=params.world_size, rank=rank, tokenizer=tokenizer)

        train(rank, device, logger, params, train_dataloader, val_dataloader=val_dataloader, tokenizer=tokenizer,
              verbose=params.verbose, pivot=True, pivot_pair_ind=params.pivot_inds)

    else:
        raise NotImplementedError

    # end wanb process to avoid hanging
    if rank == 0 and params.wandb:
        wandb.finish()

def run_distributed(params):
    params.world_size = params.gpus * params.nodes
    try:
        os.environ['MASTER_ADDR']
        os.environ['MASTER_PORT']
    except KeyError:
        print('Missing environment variable.')
        sys.exit(1)
    mp.spawn(main, nprocs=params.gpus, args=(params,))

if __name__ == "__main__":

    args = train_parser.parse_args()

    # Loader can also take in any dictionary of parameters
    params = Loader(args, check_custom=True)
    if params.distributed:
        run_distributed(params)
    else:
        params.world_size = params.gpus * params.nodes
        main(0, params)
