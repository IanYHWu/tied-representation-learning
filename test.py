"""
Inference Loop for MNMT
"""

import torch
import torch.nn.functional as F
import time
from tokenizers import Tokenizer
from models import base_transformer
from models import initialiser
from common import data_logger as logging
from common.metrics import BLEU
from common import preprocess
from common.test_arguments import test_parser
from hyperparams.loader import Loader
from common.utils import to_devices, accuracy_fn, mask_after_stop, get_all_directions

def greedy_search(x, y, y_tar, model, enc_mask=None):
    """Inference loop taking the most probable token at each step."""
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

    return torch.cat(y_pred, dim=1)

def beam_search(x, y, y_tar, model, enc_mask=None, beam_length=2):
    """Inference loop exploring the n most probable paths at each step."""
    x_enc = model.encoder(x, enc_mask)
    decode = lambda y : F.log_softmax(model.final_layer(model.decoder(y, x_enc, None, None)[0]), dim=-1)

    # initial beams and probabilities
    y = y.unsqueeze(1).repeat(1, beam_length, 1)
    log_p = torch.zeros(y.size()[0], beam_length).to(y.device) # (batch, 1)

    for t in range(y_tar.size(1)):
        with torch.no_grad():

            # expand beams
            output = decode(y.reshape(-1, y.size()[-1]))[:, -1, :] # (batch*beam, vocab_size)
            new_log_p, new_tokens = torch.topk(output, beam_length, dim=-1) # (batch*beam, beam)

            # get probability of each beam and trim beams
            log_p = log_p.unsqueeze(-1) + new_log_p.reshape(-1, beam_length, beam_length)
            log_p, indices = torch.topk(log_p.reshape(-1, beam_length * beam_length), beam_length, dim=-1)

            # get the new tokens for each beam (batch, beam)
            new_tokens = torch.gather(new_tokens.reshape(-1, beam_length*beam_length), 1, indices)

            # get the new beams
            y_ = y.unsqueeze(2).repeat(1, 1, beam_length, 1).reshape(-1, beam_length*beam_length, y.size()[-1])
            new_y = torch.gather(y_, 1, indices.unsqueeze(-1).repeat(1, 1, y_.size()[-1])) # (batch, beam, seq_len)

            y = torch.cat([new_y, new_tokens.unsqueeze(-1)], dim=-1) # (batch, beam, seq_len + 1)

    # get the beam with the highest log prob
    best_beam = log_p.argmax(-1, keepdim=True) # (batch,)
    y = torch.gather(y, 1, best_beam.unsqueeze(-1).repeat(1, 1, y_pred.size()[-1])) # (batch, tar_len+1)

    return y[:, 0, 1:] # (batch, tar_len)


def inference_step(x, y, model, logger, tokenizer, device, bleu, teacher_forcing=False, beam_length=1):
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

        batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()
        bleu(torch.argmax(y_pred, axis=-1), y_tar)

        logger.log_examples(y_tar, torch.argmax(y_pred, axis=-1), tokenizer)

    else:
        # Retrieve the start of sequence token and the target translation
        y, y_tar = y[:, 0].unsqueeze(-1), y[:, 1:]
        enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_tar)

        # devices
        x, y_tar, enc_mask = to_devices((x, y_tar, enc_mask), device)

        # inference
        model.eval()
        if beam_length == 1:
            y_pred = greedy_search(x, y, y_tar, model, enc_mask=enc_mask)
        else:
            y_pred = beam_search(x, y, y_tar, model, enc_mask=enc_mask, beam_length=beam_length)

        # loss and metrics
        batch_acc = 0
        # batch_acc = accuracy_fn(y_pred, y_tar)
        bleu(y_pred, y_tar)

        logger.log_examples(y_tar, y_pred, tokenizer)

    return batch_acc


def test(device, params, test_dataloader, tokenizer):
    """Test loop"""

    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location + '/' + params.name)

    model = initialiser.initialise_model(train_params, device)
    model = logging.load_checkpoint(logger.checkpoint_path, device, model)

    multi = False
    if len(params.langs) > 2:
        assert tokenizer is not None
        multi = True
        add_targets = preprocess.AddTargetTokens(params.langs, tokenizer)

    test_batch_accs = []
    bleu = BLEU()
    bleu.set_excluded_indices([0, 2])

    test_acc = 0.0
    start_ = time.time()

    print("Now testing")
    for i, data in enumerate(test_dataloader):

        if multi:
            x, y, y_lang = get_all_directions(data, params.langs)
            x = add_targets(x, y_lang)
        else:
            x, y = data

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


def main(params):
    """ Loads the dataset and trains the model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(params.langs) == 2:
        # bilingual translation

        try: # check for existing tokenizers
            tokenizers = [Tokenizer.from_file(params.location+'/'+lang+'_tokenizer.json') for lang in langs]
        except:
            tokenizers = None

        train_dataloader, val_dataloader, test_dataloader, tokenizers = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset,
            tokenizer=tokenizers, multi=False)

        test(device, params, test_dataloader, tokenizers)

    else:
        # multilingual translation

        try: # check for existing tokenizers
            tokenizer = Tokenizer.from_file(params.location + '/multi_tokenizer.json')
        except:
            tokenizer = None

        train_dataloader, val_dataloader, test_dataloader, tokenizer = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset,
            tokenizer=tokenizer, multi=True)

        test(device, params, test_dataloader, tokenizer)


if __name__ == "__main__":
    args = test_parser.parse_args()

    # Loader can also take in any dictionary of parameters
    params = Loader(args, check_custom=True)
    main(params)
