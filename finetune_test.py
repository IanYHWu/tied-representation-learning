""" Test a finetuned model. """
import torch
import torch.nn.functional as F 
import numpy as np
import pandas as pd
import wandb
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, MBartConfig
from datasets import load_dataset
from itertools import combinations
import time

from common.preprocess import pad_sequence, filter_languages
from common.utils import accuracy_fn, to_devices
from common.metrics import BLEU
from common import data_logger as logging
from hyperparams.schedule import WarmupDecay
from finetune import LANG_CODES

from common.preprocess import detokenize
from common.utils import mask_after_stop


def main(params):
    """ Evaluates a finetuned model on the test or validation dataset."""

    # load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    config = MBartConfig.from_pretrained("facebook/mbart-large-50")
    model = MBartForConditionalGeneration(config).to(device)
    checkpoint_location = params.location+'/'+params.name+'/checkpoint/checkpoint'
    model, _, _, _ = logging.load_checkpoint(checkpoint_location, device, model)

    def pipeline(dataset, langs, batch_size, max_len):

        cols = ['input_ids_' + l for l in langs]

        def tokenize_fn(example):
            """apply tokenization"""
            l_tok = []
            for lang in langs:
                encoded = tokenizer.encode(example[lang])
                encoded[0] = tokenizer.lang_code_to_id[LANG_CODES[lang]]
                l_tok.append(encoded)
            return {'input_ids_' + l: tok for l, tok in zip(langs, l_tok)}

        def pad_seqs(examples):
            """Apply padding"""
            ex_langs = list(zip(*[tuple(ex[col] for col in cols) for ex in examples]))
            ex_langs = tuple(pad_sequence(x, batch_first=True, max_len=max_len) for x in ex_langs)
            return ex_langs

        dataset = filter_languages(dataset, langs)
        dataset = dataset.map(tokenize_fn)
        dataset.set_format(type='torch', columns=cols)
        num_examples = len(dataset)
        print('-'.join(langs) + ' : {} examples.'.format(num_examples))
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                collate_fn=pad_seqs)
        return dataloader, num_examples

    # load data
    if params.split == 'val':
        test_dataset = load_dataset('ted_multi', split='validation')
    elif params.split == 'test':
        test_dataset = load_dataset('ted_multi', split='test')
    elif params.split == 'combine':
        test_dataset = load_dataset('ted_multi', split='validation+test')
    else:
        raise NotImplementedError

    # preprocess splits for each direction
    test_dataloaders = {}
    for l1, l2 in combinations(params.langs, 2):
        test_dataloaders[l1+'-'+l2], _ = pipeline(test_dataset, [l1, l2], params.batch_size, params.max_len)

    # evaluate the model
    def evaluate(x, y, y_code, bleu):
        y_inp, y_tar = y[:,:-1].contiguous(), y[:,1:].contiguous()
        enc_mask = (x != 0)
        x, y_inp, y_tar, enc_mask = to_devices(
          (x, y_inp, y_tar, enc_mask), device)
        
        model.eval()
        y_pred = model.generate(input_ids=x, decoder_start_token_id=y_code,
            attention_mask=enc_mask, max_length=x.size(1)+1,
            num_beams=params.num_beams, length_penalty=params.length_penalty,
            early_stopping=True)
        bleu(y_pred[:, 1:], y_tar)

    test_results = {}
    for direction, loader in test_dataloaders.items():
        alt_direction = '-'.join(reversed(direction.split('-')))
        bleu1, bleu2 = BLEU(), BLEU()
        bleu1.set_excluded_indices([0, 2])
        bleu2.set_excluded_indices([0, 2])
        x_code = tokenizer.lang_code_to_id[LANG_CODES[direction.split('-')[0]]]
        y_code = tokenizer.lang_code_to_id[LANG_CODES[direction.split('-')[-1]]]

        start_ = time.time()
        for i, (x, y) in enumerate(loader):
            if params.test_batches is not None:
                if i > params.test_batches:
                    break

            evaluate(x, y, y_code, bleu1)
            if not params.single_direction:
                evaluate(y, x, x_code, bleu2)
            if i % params.verbose == 0:
                bl1, bl2 = bleu1.get_metric(), bleu2.get_metric()
                print('Batch {} Bleu1 {:.4f} Bleu2 {:.4f} in {:.4f} secs per batch'.format(
                    i, bl1, bl2, (time.time() - start_)/(i+1)))

        bl1, bl2 = bleu1.get_metric(), bleu2.get_metric()
        test_results[direction] = [bl1]
        test_results[alt_direction] = [bl2]
        print(direction, bl1, bl2)

    # save test_results
    pd.DataFrame(test_results).to_csv(params.location+'/'+params.name+'/test_results.csv', index=False)


class ExamplesLogger:

    def __init__(self, params):
        self.params = params
        self.input_examples = []
        self.target_examples = []
        self.pred_examples = []

    def log_examples(self, input_batch, target_batch, prediction_batch, tokenizer):
        prediction_batch = mask_after_stop(prediction_batch, stop_token=2)
        if isinstance(tokenizer, list):
            inp_tokenizer = tokenizer[0]
            out_tokenizer = tokenizer[1]
        else:
            inp_tokenizer = tokenizer
            out_tokenizer = tokenizer
        det_input = str(detokenize(input_batch, inp_tokenizer, batch=False)[0])
        det_target = str(detokenize(target_batch, out_tokenizer, batch=False)[0])
        det_pred = str(detokenize(prediction_batch, out_tokenizer, batch=False)[0])

        self.target_examples.append(det_target)
        self.pred_examples.append(det_pred)
        self.input_examples.append(det_input)

    def dump_examples(self):
        with open(self.params.location+'/'+self.params.name + '_examples.txt', 'w') as f:
            for inp, pred, target in zip(self.input_examples, self.pred_examples, self.target_examples):
                f.write("Input: {} \n \n".format(inp))
                f.write("Target: {} \n \n".format(target))
                f.write("Prediction: {} \n \n".format(pred))
                f.write("---------------------------------- \n \n")


if __name__ == '__main__':

    from common.finetune_arguments import parser
    params = parser.parse_args()
    main(params)

