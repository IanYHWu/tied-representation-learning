"""
Training Loop for MNMT
"""

import numpy as np
import torch
import models.base_transformer as base_transformer
import utils.preprocess as preprocess
import utils.metrics as metrics


class Cfg:
    langs = ['en', 'fr']
    vocab_size = 2000
    batch_size = 32
    layers = 2
    heads = 4
    dff = 128
    d_model = 32
    max_pe = 1000
    dropout = 0.1
    epochs = 20


def train_step(x, y, model, optimizer, criterion):
    # get masks and targets
    y_inp, y_tar = y[:, :-1], y[:, 1:]
    enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)

    # forward
    model.train()
    y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)
    loss = criterion(y_pred.permute(0, 2, 1), y_tar)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # metrics
    batch_loss = loss.item()
    batch_acc = (torch.argmax(y_pred.detach(), axis=-1) == y_tar).numpy().mean()

    return batch_loss, batch_acc


def val_step(x, y, model, criterion):
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


def train(epochs, train_loader, val_loader=None):
    """Training Loop"""
    model = base_transformer.Transformer(Cfg.layers, Cfg.heads, Cfg.dff,
                                    Cfg.d_model, Cfg.vocab_size,
                                    Cfg.vocab_size, Cfg.max_pe,
                                    Cfg.max_pe, rate=Cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        val_epoch_bleu = 0.0
        for i, (x, y) in enumerate(train_loader):

            batch_loss, batch_acc = train_step(x, y, model, optimizer, criterion)

            train_epoch_loss += (batch_loss - train_epoch_loss) / (i + 1)
            train_epoch_acc += (batch_acc - train_epoch_acc) / (i + 1)

            if i % 50 == 0:
                print('Batch {} Training Loss {:.4f} Accuracy {:.4f}'.format(i, train_epoch_loss, train_epoch_acc))

        print('Epoch {} Training Loss {:.4f} Accuracy {:.4f} \n'.format(epoch, train_epoch_loss, train_epoch_acc))

        if val_loader is not None:
            for i, (x, y) in enumerate(val_loader):
                batch_loss, batch_acc, batch_bleu = val_step(x, y, model, criterion)

                val_epoch_loss += (batch_loss - val_epoch_loss) / (i + 1)
                val_epoch_acc += (batch_acc - val_epoch_acc) / (i + 1)
                val_epoch_bleu += (batch_bleu - val_epoch_bleu) / (i + 1)

                if i % 50 == 0:
                    print('Batch {} Val Loss {:.4f} Accuracy {:.4f} Bleu {:.4f}'.
                          format(i, val_epoch_loss, val_epoch_acc, val_epoch_bleu))

            print(
                'Epoch {} Val Loss {:.4f} Accuracy {:.4f} Bleu {:.4f}\n'.
                    format(epoch, val_epoch_loss, val_epoch_acc, val_epoch_bleu))


if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = \
    preprocess.load_and_preprocess(Cfg.langs, Cfg.batch_size, Cfg.vocab_size, "ted_multi")

    train(epochs=30, train_loader=train_dataloader, val_loader=val_dataloader)
