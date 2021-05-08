"""
Inference Loop for MNMT
"""

import torch
import torch.nn.functional as F
import numpy as np 
import time
from tokenizers import Tokenizer
from common.preprocess import detokenize, tokenize
from models import base_transformer
from models import initialiser
from common import data_logger as logging
from common.metrics import BLEU
from common import preprocess
from common.test_arguments import test_parser
from hyperparams.loader import Loader
from common.utils import to_devices, accuracy_fn, mask_after_stop, get_all_directions, get_pairs, get_directions


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


def single_beam_search(x, y, y_tar, model, enc_mask=None, beam_length=2, alpha=0.0, beta=0.0):
    """
    x : (seq_len)
    y : ([]) tensor of start token
    max_len maximum length greater than the length of x to consider
    """

    if enc_mask is not None:
        enc_mask = enc_mask.unsqueeze(0)

    x_enc = model.encoder(x.unsqueeze(0), enc_mask) # (1, seq_len, d_model)
    x_enc = x_enc.repeat(beam_length, 1, 1) # (beam, seq_len, d_model)

    y = y.reshape(1, 1).repeat(beam_length, 1) # (beam, 1)
    log_p = torch.zeros(beam_length).to(y.device)

    attn_block = 'decoder_layer' + str(len(model.decoder.dec_layers)) + '_block2'

    for t in range(y_tar.size(0)):
        with torch.no_grad():

            # expand beams
            y_dec, attn_weights = model.decoder(y, x_enc, None, None)
            y_pred = F.log_softmax(model.final_layer(y_dec), dim=-1)[:, -1, :] # (beam, vocab)
            new_log_p = (log_p.unsqueeze(-1) + y_pred)

            # compute beam score by s = logP / lp + cp

            # cp
            attn_weights = attn_weights[attn_block].mean(1).sum(1) # (beam, x_len)
            cp = beta * torch.log(torch.clip(attn_weights, low=1.0)).sum(-1) # (beam,)

            # lp
            lp = ((y_tar.size(0) + 5) ** alpha) / (6 ** alpha) # (beam, vocab)

            # s
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
    

def beam_search(x, y, y_tar, model, enc_mask=None, beam_length=2):
    preds = []
    for i in range(x.size(0)):
        enc_mask_i = enc_mask[i] if enc_mask is not None else None
        preds.append(single_beam_search(
            x[i], y[i], y_tar[i], model,
            enc_mask=enc_mask_i, beam_length=beam_length)
        )
    return torch.stack(preds, dim=0)


def inference_step(x, y, model, logger, tokenizer, device, bleu=None,
                   teacher_forcing=False, pivot_mode=False, beam_length=1):
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
            y_pred = beam_search(x, y, y_tar, model, enc_mask=enc_mask, beam_length=beam_length)

        if not pivot_mode:
            batch_acc = 0
            if bleu is not None:
                bleu(y_pred, y_tar)
            logger.log_examples(x, y_tar, y_pred, tokenizer)
            return batch_acc
        else:
            return y_pred


def test(device, params, test_dataloader, tokenizer, verbose=50):
    """Test loop"""

    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location + '/' + params.name)

    model = initialiser.initialise_model(train_params, device)
    model, _, _, _ = logging.load_checkpoint(logger.checkpoint_path, device, model)

    test_batch_accs = []
    bleu = BLEU()
    bleu.set_excluded_indices([0, 2])

    test_acc = 0.0
    start_ = time.time()

    print(params.__dict__)
    print("Now testing")
    for i, data in enumerate(test_dataloader):

        x, y = data
        test_batch_acc = inference_step(x, y, model, logger, tokenizer, device, bleu=bleu,
                                        teacher_forcing=params.teacher_forcing,
                                        beam_length=params.beam_length)
        test_batch_accs.append(test_batch_acc)

        test_acc += (test_batch_acc - test_acc) / (i + 1)
        curr_bleu = bleu.get_metric()

        if verbose is not None:
            if i % verbose == 0:
                print('Batch {} Accuracy {:.4f} Bleu {:.4f} in {:.4f} s per batch'.format(
                    i, test_acc, curr_bleu, (time.time() - start_) / (i + 1)))

    test_bleu = bleu.get_metric()
    direction = params.langs[0] + '-' + params.langs[1]
    logger.log_results([direction, test_acc, test_bleu])
    logger.dump_examples()


def multi_test(device, params, test_dataloader, tokenizer, verbose=50):
    """Test for multilingual translation. Evaluates on all possible translation directions."""

    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location + '/' + params.name)

    model = initialiser.initialise_model(train_params, device)
    model, _, _, _ = logging.load_checkpoint(logger.checkpoint_path, device, model)

    assert tokenizer is not None
    add_targets = preprocess.AddTargetTokens(params.langs, tokenizer)
    pair_accs = {s+'-'+t : 0.0 for s, t in get_pairs(params.langs)}
    pair_bleus = {}
    for s, t in get_pairs(params.langs, excluded=params.excluded):
        _bleu = BLEU()
        _bleu.set_excluded_indices([0, 2])
        pair_bleus[s+'-'+t] = _bleu

    test_acc = 0.0
    start_ = time.time()

    print(params.__dict__)
    print("Now testing")
    for i, data in enumerate(test_dataloader):

        data = get_directions(data, params.langs, excluded=params.excluded)
        for direction, (x, y, y_lang) in data.items():
            x = add_targets(x, y_lang)
            bleu = pair_bleus[direction]
            test_batch_acc = inference_step(x, y, model, logger, tokenizer, device, bleu=bleu,
                                            teacher_forcing=params.teacher_forcing,
                                            beam_length=params.beam_length)
            pair_accs[direction] += (test_batch_acc - pair_accs[direction]) / (i + 1)

        # report the mean accuracy and bleu accross directions
        if verbose is not None:
            test_acc += (np.mean([v for v in pair_accs.values()]) - test_acc) / (i + 1)
            curr_bleu = np.mean([bleu.get_metric() for bleu in pair_bleus.values()])
            if i % verbose == 0:
                print('Batch {} Accuracy {:.4f} Bleu {:.4f} in {:.4f} s per batch'.format(
                    i, test_acc, curr_bleu, (time.time() - start_) / (i + 1)))

    directions = [d for d in pair_bleus.keys()]
    test_accs = [pair_accs[d] for d in directions]
    test_bleus = [pair_bleus[d].get_metric() for d in directions]
    logger.log_results([directions, test_accs, test_bleus])
    logger.dump_examples()


def pivot_test(device, params, test_dataloader_1, test_dataloader_2, tokenizer_1, tokenizer_2, verbose=50):
    logger = logging.TestLogger(params)
    logger.make_dirs()
    train_params = logging.load_params(params.location + '/' + params.name)

    model_1 = initialiser.initialise_model(train_params, device)
    state_1 = torch.load(params.pivot_model_1, map_location=device)
    model_1.load_state_dict(state_1['model_state_dict'])

    model_2 = initialiser.initialise_model(train_params, device)
    state_2 = torch.load(params.pivot_model_2, map_location=device)
    model_2.load_state_dict(state_2['model_state_dict'])

    test_batch_accs = []
    bleu = BLEU()
    bleu.set_excluded_indices([0, 2])

    test_acc = 0.0
    start_ = time.time()

    for i, (data_1, data_2) in enumerate(zip(test_dataloader_1, test_dataloader_2)):
        x_1, y_1 = data_1
        x_2, y_2 = data_2

        y_pred_1 = inference_step(x_1, y_1, model_1, logger, tokenizer_1, device,
                                  teacher_forcing=params.teacher_forcing, pivot_mode=True)
        y_pred_det = detokenize(y_pred_1, tokenizer_1[1])
        y_pred_tok = tokenize(y_pred_det, tokenizer_2[0])
        test_batch_acc = inference_step(y_pred_tok, y_2, model_2, logger, tokenizer_2, device,
                                        teacher_forcing=params.teacher_forcing, pivot_mode=False, bleu=bleu)

        test_batch_accs.append(test_batch_acc)
        test_acc += (test_batch_acc - test_acc) / (i + 1)
        curr_bleu = bleu.get_metric()

        if verbose is not None:
            if i % verbose == 0:
                print('Batch {} Accuracy {:.4f} Bleu {:.4f} in {:.4f} s per batch'.format(
                    i, test_acc, curr_bleu, (time.time() - start_) / (i + 1)))

    test_bleu = bleu.get_metric()
    direction = "pivot"
    logger.log_results([direction, test_acc, test_bleu])
    logger.dump_examples()


def main(params):
    """ Loads the dataset and trains the model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(params.langs) == 2:
        # bilingual translation
        # check for existing tokenizers
        if params.tokenizer is not None:
            tokenizers = [Tokenizer.from_file('pretrained/' + tok + '.json') for tok in params.tokenizer]
        else:
            try:
                tokenizers = [Tokenizer.from_file(params.location + '/' + lang + '_tokenizer.json') for lang in
                          params.langs]
            except:
                tokenizers = None

        train_dataloader, val_dataloader, test_dataloader, tokenizers = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset,
            tokenizer=tokenizers, multi=False)

        test(device, params, test_dataloader, tokenizers, verbose=params.verbose)

    elif len(params.langs) > 2 and not params.pivot:
        # multilingual translation
        #  check for existing tokenizers
        if params.tokenizer is not None:
            tokenizer = Tokenizer.from_file('pretrained/' + params.tokenizer + '.json')
        else:
            try:
                tokenizer = Tokenizer.from_file(params.location + '/multi_tokenizer.json')
            except:
                tokenizer = None

        train_dataloader, val_dataloader, test_dataloader, tokenizer = preprocess.load_and_preprocess(
            params.langs, params.batch_size, params.vocab_size, params.dataset,
            tokenizer=tokenizer, multi=True)

        multi_test(device, params, test_dataloader, tokenizer, verbose=params.verbose)

    elif len(params.langs) > 2 and params.pivot:
        try:
            tokenizer_1_1 = Tokenizer.from_file(params.pivot_tokenizer_path_1_1)
            tokenizer_1_2 = Tokenizer.from_file(params.pivot_tokenizer_path_1_2)
            tokenizer_2_1 = Tokenizer.from_file(params.pivot_tokenizer_path_2_1)
            tokenizer_2_2 = Tokenizer.from_file(params.pivot_tokenizer_path_2_2)
            tokenizer_1 = [tokenizer_1_1, tokenizer_1_2]
            tokenizer_2 = [tokenizer_2_1, tokenizer_2_2]
        except:
            tokenizer_1 = None
            tokenizer_2 = None

        test_dataloader_1, test_dataloader_2 = preprocess.pivot_load_and_preprocess(params.langs,
                                                                                    params.batch_size,
                                                                                    params.dataset,
                                                                                    tokenizer_1=tokenizer_1,
                                                                                    tokenizer_2=tokenizer_2)

        pivot_test(device, params, test_dataloader_1, test_dataloader_2, tokenizer_1, tokenizer_2,
                   verbose=params.verbose)


if __name__ == "__main__":
    args = test_parser.parse_args()

    # Loader can also take in any dictionary of parameters
    params = Loader(args, check_custom=True)
    main(params)
