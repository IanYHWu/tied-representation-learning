""" Finetune MBart for MNMT on given langauges. """
import torch
import numpy as np
import pandas as pd
import wandb
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from datasets import load_dataset
from itertools import combinations

from common.preprocess import pad_sequence, filter_languages
from common.utils import accuracy_fn, to_devices
from common.metrics import BLEU
from common import data_logger as logging


# mapping ted to mBart lang codes
LANG_CODES = {
    'ar' : 'ar_AR', # Arabic
    'cs' : 'cs_CZ', # Czech
    'en' : 'en_XX', # English
    'fr' : 'fr_XX', # French
    'de' : 'de_DE', # German
    'he' : 'he_IL', # Hebrew
    'id' : 'id_ID', # Indonesian
    'ja' : 'ja_XX', # Japense
    'ko' : 'ko_KR', # Korean
    'pl' : 'pl_PL', # Polish
    'pt' : 'pt_XX', # Portugese
    'ro' : 'ro_RO', # Romanian
    'ru' : 'ru_RU', # Russian
    'tr' : 'tr_TR', # Turkish
    'uk' : 'uk_UA', # Ukranian
    'vi' : 'vi_VN', # Vietnamese
    'zh' : 'zh_CN' # Chinese
}


def main(params):
    """ Finetunes the mBart50 model on some languages and
    then evaluates the BLEU score for each direction."""

    if params.wandb:
        wandb.init(project='mnmt', entity='nlp-mnmt-project', group='finetuning',
            config={k: v for k, v in params.__dict__.items() if isinstance(v, (float, int, str, list))})

    new_root_path = params.location
    new_name = params.name
    logger = logging.TrainLogger(params)
    logger.make_dirs()
    logger.save_params()

    # load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    from common.preprocess import pad_sequence, filter_languages
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
    dataset = load_dataset('ted_multi')
    train_dataset = dataset['train']
    #val_dataset = dataset['validation']
    test_dataset = dataset['test']

    # preprocess splits for each direction
    num_train_examples = {}
    train_dataloaders, val_dataloaders, test_dataloaders = {}, {}, {}
    for l1, l2 in combinations(params.langs, 2):
        train_dataloaders[l1+'-'+l2], num_train_examples[l1+'-'+l2] = pipeline(
            train_dataset, [l1, l2], params.batch_size, params.max_len)
        #val_dataloaders[l1+'-'+l2], _ = pipeline(val_dataset, [l1, l2], params.batch_size, params.max_len)
        test_dataloaders[l1+'-'+l2], _ = pipeline(test_dataset, [l1, l2], params.batch_size, params.max_len)

    # print dataset sizes
    for direction, num in num_train_examples.items():
        print(direction, ': {} examples.'.format(num))

    # train the model
    def train_step(x, y):

        y_inp, y_tar = y[:,:-1].contiguous(), y[:,1:].contiguous()
        enc_mask, dec_mask = (x != 0), (y_inp != 0)

        x, y_inp, y_tar, enc_mask, dec_mask = to_devices(
          (x, y_inp, y_tar, enc_mask, dec_mask), device)

        model.train()
        output = model(input_ids=x, decoder_input_ids=y_inp,
                   labels=y_tar, attention_mask=enc_mask,
                   decoder_attention_mask=dec_mask)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()

        accuracy = accuracy_fn(output.logits, y_tar)

        return output.loss.item(), accuracy.item()

    # prepare iterators
    iterators = {direction: iter(loader) for direction, loader in train_dataloaders.items()}
    directions, num_examples = list(num_train_examples.keys()), np.array(list(num_train_examples.values()))
    dir_dist = (num_examples ** params.temp) / ((num_examples ** params.temp).sum())

    #train
    losses, accs = [], []
    for i in range(params.train_steps):

        # sample a direction
        data = next(iterators[directions[int(np.random.choice(len(num_examples), p=dir_dist))]])
        if np.random.rand() > 0.5:
          x, y = data 
        else:
          y, x = data 
           
        # train on the direction
        loss, acc = train_step(x, y)
        losses.append(loss)
        accs.append(acc)

        if i % params.verbose == 0:
            print('Batch {} Loss {:.4f} Acc {:.4f}'.format(
                i, np.mean(losses[-params.verbose:]), np.mean(accs[-params.verbose:])))
        if params.wandb:
            wandb.log({'train_loss':loss, 'train_acc':acc})

    # save results
    if params.save:
        logger.save_model(params.train_steps, model, optimizer)
    
    logger.log_results([np.mean(losses), np.mean(accs)])

    # evaluate the model
    def evaluate(x, y, y_code, bleu):
        y_inp, y_tar = y[:,:-1].contiguous(), y[:,1:].contiguous()
        enc_mask, dec_mask = (x != 0), (y_inp != 0)
        x, y_inp, y_tar, enc_mask, dec_mask = to_devices(
          (x, y_inp, y_tar, enc_mask, dec_mask), device)
        
        model.eval()
        y_pred = model.generate(input_ids=x, decoder_start_token_id=y_code,
            attention_mask=enc_mask, max_length=params.max_len+1)
        bleu(y_pred[:,1:], y_tar)

    test_results = {}
    for direction, loader in test_dataloaders.items():
        alt_direction = '-'.join(reversed(direction.split('-')))
        bleu1, bleu2 = BLEU(), BLEU()
        bleu1.set_excluded_indices([0, 2])
        bleu2.set_excluded_indices([0, 2])
        x_code = tokenizer.lang_code_to_id(LANG_CODES[direction.split('-')[0]])
        y_code = tokenizer.lang_code_to_id(LANG_CODES[direction.split('-')[-1]])

        for x, y in loader:
            evaluate(x, y, y_code, bleu1)
            evaluate(y, x, x_code, bleu2)

        test_results[direction] = [bleu1.get_metric()]
        test_results[alt_direction] = [bleu2.get_metric()]

    # save test_results
    pd.DataFrame(test_results).to_csv(logger.root_path + '/test_results.csv', index=False)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--name',
        required=True, type=str,
        help='Name of run.'
    )

    parser.add_argument('--location',
        default='..', type=str,
        help='File path for saving results.'
    )
    parser.add_argument('--langs',
        default=['en', 'fr'], nargs='+',
        help='Ted language codes for languages to train on.'
    )
    parser.add_argument('--batch_size',
        default=6, type=int,
        help='Dataloader batch size.'
    )
    parser.add_argument('--max_len',
        default=100, type=int,
        help='Maximum length of sequences.'
    )
    parser.add_argument('--train_steps',
        default=5000, type=int,
        help='Total number of training steps.'
    )
    parser.add_argument('--lr',
        default=1e-4, type=float,
        help='Finetuning learning rate.'
    )
    parser.add_argument('--temp',
        default=1.0, type=float,
        help='Sampling temperature for relative language sizes.'
    )
    parser.add_argument('--verbose',
        default=20, type=int,
        help='Frequency to print during training.'
    )
    parser.add_argument('--wandb',
        action='store_true',
        help='wether to log results to wandb.'
    )
    parser.add_argument('--save',
        action='store_true',
        help='wether to save model after training.'
    )

    params = parser.parse_args()
    #main(params)

