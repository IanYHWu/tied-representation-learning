"""
Inference Loop for MNMT
"""

import torch
import time
from models import base_transformer
from train import to_devices, accuracy_fn
from models import initialiser
from common import data_logger as logging
from common.metrics import compute_bleu


def inference_step(x, y, model, criterion, device):
    """
    inference step.
    x: source language
    y: target language
    """

    # Retrieve the start of sequence token and the target translation
    y, y_tar = y[:, 0].unsqueeze(-1), y[:, 1:]
    enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_tar)

    # devices
    x, y_tar, enc_mask = to_devices((x, y_tar, enc_mask), device)

    # forward
    model.eval()
    x_enc = model.encoder(x, enc_mask)
    y_pred = []  # List to store predictions made by model.

    # loop to compute output of decoder without teacher forcing
    for t in range(y_tar.size(1)):
        with torch.no_grad():
            # Compute output of all translations up until sequence position t.
            output, hidden = model.final_layer(model.decoder(y, x_enc, None, None))
            # Take most recently computed time step.
            output = output[:, -1, :].squeeze()
            # Retrieve predicted token.
            max_token = torch.argmax(output, dim=-1).unsqueeze(-1)
            y_pred.append(max_token)
            y = torch.cat([y, max_token], dim=1)

            # EOS check.
            c = 0
            for i in range(y.size(0)):
                if 2 not in y[i, :]:
                    break
                else:
                    c = c + 1

            if c == y.size(0):
                break

    # loss and metrics
    y_pred = torch.cat(y_pred, dim=1)
    loss = criterion(y_pred.permute(0, 2, 1), y_tar)
    batch_loss = loss.item()
    batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()
    batch_bleu = compute_bleu(y_tar, y_pred)

    return batch_loss, batch_acc, batch_bleu


def test(device, params, test_dataloader):
    """Test loop"""

    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location)

    model = initialiser.initialise_model(train_params, device)
    model = logging.load_checkpoint(logger.checkpoint_path, model)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    test_batch_losses, test_batch_accs, test_batch_bleus = [], [], []

    test_loss = 0.0
    test_acc = 0.0
    test_bleu = 0.0
    start_ = time.time()

    for i, (x, y) in enumerate(test_dataloader):

        test_batch_loss, test_batch_acc, test_batch_bleu = inference_step(x, y, model, criterion, device)

        test_batch_losses.append(test_batch_loss)
        test_batch_accs.append(test_batch_acc)
        test_batch_bleus.append(test_batch_bleu)

        test_loss += (test_batch_loss - test_loss) / (i + 1)
        test_acc += (test_batch_acc - test_acc) / (i + 1)
        test_bleu += (test_batch_bleu - test_bleu) / (i + 1)

        if i % 50 == 0:
            print('Batch {} Loss {:.4f} Accuracy {:.4f} Bleu {:.4f} in {:.4f} s per batch'.format(
                i, test_loss, test_acc, test_bleu, (time.time() - start_) / (i + 1)))

    logger.log_results([params.langs, test_loss, test_acc, test_bleu])

