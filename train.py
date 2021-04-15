"""
Training Loop for MNMT
"""

import torch
import time

import models.base_transformer as base_transformer
import models.initialiser as initialiser
from common import preprocess
from common.train_arguments import train_parser
from common import data_logger as logging
from hyperparams.loader import Loader
from hyperparams.schedule import WarmupDecay
from common.metrics import BLEU
from common.utils import to_devices, accuracy_fn, loss_fn, sample_direction


def train_step(x, y, model, criterion, optimizer, scheduler, device):
    # get masks and targets
    y_inp, y_tar = y[:, :-1], y[:, 1:]
    enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)

    # devices
    x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask = to_devices(
        (x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask),
        device)

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
    batch_loss = loss.cpu().item()
    batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()

    return batch_loss, batch_acc


def val_step(x, y, model, criterion, bleu, device):
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
        params.epochs += params.add_epochs
        logger = logging.TrainLogger(params)
        logger.make_dirs()
    else:
        logger = logging.TrainLogger(params)
        logger.make_dirs()
    logger.save_params()
    return logger

def train(device, logger, params, train_dataloader, val_dataloader=None, tokenizer=None):
    """Training Loop"""

    multi = False
    if len(params.langs) > 2:
        assert tokenizer is not None
        multi = True
        add_targets = preprocess.AddTargetTokens(params.langs, tokenizer)

    model = initialiser.initialise_model(params, device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = WarmupDecay(optimizer, params.warmup_steps, params.d_model, lr_scale=params.lr_scale)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    epoch = 0
    if params.checkpoint:
        model, optimizer, epoch = logging.load_checkpoint(logger.checkpoint_path, model, optimizer)

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
            else:
                x, y = data

            batch_loss, batch_acc = train_step(x, y, model, criterion, optimizer, scheduler, device)

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
            bleu = BLEU()
            bleu.set_excluded_indices([0, 2])
            for i, data in enumerate(val_dataloader):
                if multi:
                    # sample a tranlsation direction and add target tokens
                    (x, y), (x_lang, y_lang) = sample_direction(data, params.langs)
                    x = add_targets(x, y_lang)
                else:
                    x, y = data

                batch_loss, batch_acc = val_step(x, y, model, criterion, bleu, device)
                val_epoch_loss += (batch_loss - val_epoch_loss) / (i + 1)
                val_epoch_acc += (batch_acc - val_epoch_acc) / (i + 1)

            val_epoch_losses.append(val_epoch_loss)
            val_epoch_accs.append(val_epoch_acc)
            val_bleu = bleu.get_metric()

            print('Epoch {} Loss {:.4f} Accuracy {:.4f} Val Loss {:.4f} Val Accuracy {:.4f} Val Bleu {:.4f}'
                  ' in {:.4f} secs \n'.format(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, val_bleu,
                                              time.time() - start_))
        else:
            print('Epoch {} Loss {:.4f} Accuracy {:.4f} in {:.4f} secs \n'.format(
                epoch, epoch_loss, epoch_acc, time.time() - start_))

        epoch += 1

        logger.save_model(epoch, model, optimizer)
        logger.log_results([epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, val_bleu])

    return epoch_losses, epoch_accs, val_epoch_losses, val_epoch_accs


def main(params):
    """ Loads the dataset and trains the model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup(params)

    if len(params.langs) == 2:
        # bilingual translation

        train_dataloader, val_dataloader, test_dataloader, _ = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset, multi=False, path=logger.root_path)

        train(device, logger, params, train_dataloader, val_dataloader=val_dataloader)
    else:
        # multilingual translation

        train_dataloader, val_dataloader, test_dataloader, tokenizer = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset, multi=True, path=logger.root_path)

        train(device, logger, params, train_dataloader, val_dataloader=val_dataloader, tokenizer=tokenizer)


if __name__ == "__main__":
    args = train_parser.parse_args()

    # Loader can also take in any dictionary of parameters
    params = Loader(args, check_custom=True)
    main(params)
