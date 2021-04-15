"""
Inference Loop for MNMT
"""

import torch
import time
from models import base_transformer
from models import initialiser
from common import data_logger as logging
from common.metrics import BLEU
from common import preprocess
from common.test_arguments import test_parser
from hyperparams.loader import Loader
from common.utils import to_devices, accuracy_fn, mask_after_stop


def inference_step(x, y, model, logger, tokenizer, device, bleu=None, teacher_forcing=False, pivot_mode=False):
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

        # forward
        model.eval()
        with torch.no_grad():
            y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)

        if not pivot_mode:
            batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()
            bleu(torch.argmax(y_pred, axis=-1), y_tar)
            logger.log_examples(y_tar, torch.argmax(y_pred, axis=-1), tokenizer)

            return batch_acc
        else:
            return y_pred

    else:
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
                output = model.final_layer(model.decoder(y, x_enc, None, None)[0])
                # Take most recently computed time step.
                output = output[:, -1, :].squeeze()
                # Retrieve predicted token.
                max_token = torch.argmax(output, dim=-1).unsqueeze(-1)
                y = torch.cat([y, max_token], dim=1)
                y_pred.append(max_token)

        # loss and metrics
        y_pred = torch.cat(y_pred, dim=1)

        if not pivot_mode:
            batch_acc = 0
            # batch_acc = accuracy_fn(y_pred, y_tar)
            if bleu is not None:
                bleu(y_pred, y_tar)
            logger.log_examples(y_tar, y_pred, tokenizer)

            return batch_acc
        else:
            return y_pred


def test(device, params, test_dataloader, tokenizer):
    """Test loop"""

    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location + '/' + params.name)

    model = initialiser.initialise_model(train_params, device)
    model = logging.load_checkpoint(logger.checkpoint_path, device, model)

    test_batch_accs = []
    bleu = BLEU()
    bleu.set_excluded_indices([0, 2])

    test_acc = 0.0
    start_ = time.time()

    print(params.__dict__)

    print("Now testing")
    for i, (x, y) in enumerate(test_dataloader):

        test_batch_acc = inference_step(x, y, model, logger, tokenizer, device, bleu,
                                                         params.teacher_forcing)

        test_batch_accs.append(test_batch_acc)

        test_acc += (test_batch_acc - test_acc) / (i + 1)
        curr_bleu = bleu.get_metric()

        if i % 50 == 0:
            print('Batch {} Accuracy {:.4f} Bleu {:.4f} in {:.4f} s per batch'.format(
                i, test_acc, curr_bleu, (time.time() - start_) / (i + 1)))

    test_bleu = bleu.get_metric()
    logger.log_results([params.langs, test_acc, test_bleu])
    logger.dump_examples()


def pivot_test(device, params, test_dataloader, tokenizers):
    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location + '/' + params.name)

    model_1 = initialiser.initialise_model(train_params, device)
    model_1 = logging.load_checkpoint(logger.checkpoint_path, device, model_1)
    tokenizer_1 = tokenizers[0]

    model_2 = initialiser.initialise_model(train_params, device)
    model_2 = logging.load_checkpoint(logger.checkpoint_path, device, model_2)
    tokenizer_2 = tokenizers[1]

    test_batch_accs = []
    bleu = BLEU()
    bleu.set_excluded_indices([0, 2])

    test_acc = 0.0
    start_ = time.time()

    for i, data in enumerate(test_dataloader):
        x_1, y_1, y_2 = data[0], data[1], data[2]

        y_pred_1 = inference_step(x_1, y_1, model_1, logger, tokenizer_1, device,
                       params.teacher_forcing, pivot_mode=True)

        test_batch_acc = inference_step(y_pred_1, y_2, model_2, logger, tokenizer_2, device, bleu,
                                  params.teacher_forcing, pivot_mode=False)

        test_batch_accs.append(test_batch_acc)

        test_acc += (test_batch_acc - test_acc) / (i + 1)
        curr_bleu = bleu.get_metric()

        if i % 50 == 0:
            print('Batch {} Accuracy {:.4f} Bleu {:.4f} in {:.4f} s per batch'.format(
                i, test_acc, curr_bleu, (time.time() - start_) / (i + 1)))

    test_bleu = bleu.get_metric()
    logger.log_results([params.langs, test_acc, test_bleu])
    logger.dump_examples()


def main(params):
    """ Loads the dataset and trains the model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(params.langs) == 2:
        # bilingual translation
        train_dataloader, val_dataloader, test_dataloader, tokenizers = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset, multi=False)

        test(device, params, test_dataloader, tokenizers)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = test_parser.parse_args()

    # Loader can also take in any dictionary of parameters
    params = Loader(args, check_custom=True)
    main(params)
