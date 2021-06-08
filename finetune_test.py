""" Test a finetuned model. """
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import wandb
from transformers import MBartForConditionalGeneration, MBartConfig
import time

from common import data_logger as logging
from common.utils import to_devices
from common.metrics import BLEU
from common.data import MNMTDataModule, LANG_CODES


def main(params):
    """ Evaluates a finetuned model on the test or validation dataset."""
    pl.seed_everything(params.seed, workers=True)

    # get data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datamodule = MNMTDataModule(params.langs, params.batch_size, params.max_len, T=params.temp)
    datamodule.prepare_data()
    datamodule.setup(stage='test')
    test_dataloaders = datamodule.splits['validation'] if params.split == 'val' else datamodule.splits['test']
    tokenizer = datamodule.tokenizer

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = MBartConfig.from_pretrained("facebook/mbart-large-50")
    model = MBartForConditionalGeneration(config).to(device)
    checkpoint_location = params.location+'/'+params.name+'/checkpoint/checkpoint'
    model, _, _, _ = logging.load_checkpoint(checkpoint_location, device, model)

    # evaluate the model
    def evaluate(x, y, y_code, bleu):
        y_inp, y_tar = y[:,:-1].contiguous(), y[:,1:].contiguous()
        enc_mask = (x != 0)
        x, y_inp, y_tar, enc_mask = to_devices(
            (x, y_inp, y_tar, enc_mask), device)
        
        model.eval()
        y_pred = model.generate(input_ids=x, decoder_start_token_id=y_code,
            attention_mask=enc_mask, max_length=x.size(1)+50,
            num_beams=params.num_beams, length_penalty=params.length_penalty,
            early_stopping=True)
        bleu(y_pred[:, 1:], y_tar)

    test_results = {}
    for direction, loader in test_dataloaders.items():
        l0, l1 = direction.split('-')[0], direction.split('-')[1]
        alt_direction = l1+'-'+l0
        bleu1, bleu2 = BLEU(), BLEU()
        bleu1.set_excluded_indices([0, 2])
        bleu2.set_excluded_indices([0, 2])
        x_code = tokenizer.lang_code_to_id[LANG_CODES[direction.split('-')[0]]]
        y_code = tokenizer.lang_code_to_id[LANG_CODES[direction.split('-')[-1]]]

        start_ = time.time()
        for i, data in enumerate(loader):
            x, y = data['input_ids_'+l0], data['input_ids_'+l1]
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


if __name__ == '__main__':

    from common.finetune_arguments import parser
    params = parser.parse_args()
    main(params)

