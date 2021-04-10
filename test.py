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
from common import preprocess
from common.test_arguments import test_parser
from hyperparams.loader import Loader


def inference_step(x, y, model, logger, tokenizer, device, teacher_forcing=False):
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

        batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()
        batch_bleu = compute_bleu(y_tar, y_pred)

        logger.log_examples(y_tar, torch.argmax(y_pred, axis=-1), tokenizer)

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
        batch_acc = 0
        # batch_acc = accuracy_fn(y_pred, y_tar)
        batch_bleu = compute_bleu(y_tar, y_pred)

        logger.log_examples(y_tar, y_pred, tokenizer)

    return batch_acc, batch_bleu


def test(device, params, test_dataloader, tokenizer):
    """Test loop"""

    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location + '/' + params.name)

    model = initialiser.initialise_model(train_params, device)
    model = logging.load_checkpoint(logger.checkpoint_path, device, model)

    test_batch_accs, test_batch_bleus = [], []

    test_acc = 0.0
    test_bleu = 0.0
    start_ = time.time()

    print("Now testing")
    for i, (x, y) in enumerate(test_dataloader):

        test_batch_acc, test_batch_bleu = inference_step(x, y, model, logger, tokenizer, device, params.teacher_forcing)

        test_batch_accs.append(test_batch_acc)
        test_batch_bleus.append(test_batch_bleu)

        test_acc += (test_batch_acc - test_acc) / (i + 1)
        test_bleu += (test_batch_bleu - test_bleu) / (i + 1)

        if i % 50 == 0:
            print('Accuracy {:.4f} Bleu {:.4f} in {:.4f} s per batch'.format(
                i, test_acc, test_bleu, (time.time() - start_) / (i + 1)))

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

