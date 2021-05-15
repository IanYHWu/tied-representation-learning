""" Finetune MBart for MNMT on given langauges. """
import torch
import torch.nn.functional as F 
import numpy as np
import pandas as pd
import wandb
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from datasets import load_dataset
from itertools import combinations
import time

from common.preprocess import pad_sequence, filter_languages
from common.utils import accuracy_fn, to_devices
from common.metrics import BLEU
from common import data_logger as logging
from hyperparams.schedule import WarmupDecay


# mapping ted to mBart lang codes
LANG_CODES = {
    'ar' : 'ar_AR', # Arabic
    'az' : 'az_AZ', # Azerbaijani
    'bn' : 'bn_IN', # Bengali
    'cs' : 'cs_CZ', # Czech
    'de' : 'de_DE', # German
    'en' : 'en_XX', # English
    'es' : 'es_XX', # Spanish
    'et' : 'et_EE', # Estonian
    'fi' : 'fi_FI', # Finish
    'fr' : 'fr_XX', # French
    'gl' : 'gl_ES', # Galician
    'he' : 'he_IL', # Hebrew
    'hi' : 'hi_IN', # Hindi
    'hr' : 'hr_HR', # Croation
    'id' : 'id_ID', # Indonesian
    'it' : 'it_IT', # Italian
    'ja' : 'ja_XX', # Japense
    'ka' : 'ka_GE', # Georgian
    'kk' : 'kk_KZ', # Kazakh
    'ko' : 'ko_KR', # Korean
    'lt' : 'lt_LT', # Lithuanian
    'mk' : 'mk_MK', # Macedonian
    'mn' : 'mn_MN', # Mongolian
    'mr' : 'mr_IN', # Marathi
    'my' : 'my_MM', # Burmese
    'nl' : 'nl_XX', # Dutch
    'pl' : 'pl_PL', # Polish
    'pt' : 'pt_XX', # Portugese
    'ro' : 'ro_RO', # Romanian
    'ru' : 'ru_RU', # Russian
    'sl' : 'sl_SI', # Slovene
    'sv' : 'sv_SE', # Swedish
    'ta' : 'ta_IN', # Tamil
    'th' : 'th_TH', # Thai
    'tr' : 'tr_TR', # Turkish
    'uk' : 'uk_UA', # Ukranian
    'ur' : 'ur_PK', # Urdu
    'vi' : 'vi_VN', # Vietnamese
    'zh' : 'zh_CN', # Chinese
}


def get_direction(x, y, sample=False):
    """ Samples a training direction from two sequences
    or returns the standard direction. """
    if sample:
        if np.random.rand() > 0.5:
            return x, y
        else:
            return y, x
    else:
        return x, y


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
    optimizer = torch.optim.Adam(model.parameters())

    # scale in terms of max lr
    lr_scale = params.max_lr * np.sqrt(params.warmup_steps)
    scheduler = WarmupDecay(optimizer, params.warmup_steps, 1, lr_scale=lr_scale)

    # set dropout
    model.config.dropout = params.dropout 
    model.config.attention_dropout = params.dropout

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

    def freeze_layers(layers, unfreeze=False):
        for n in layers:
            for parameter in model.model.encoder.layers[n].parameters():
                parameter.requires_grad = unfreeze

    # train the model
    _target = torch.tensor(1.0)
    def train_step(x, y, aux=False):

        y_inp, y_tar = y[:,:-1].contiguous(), y[:,1:].contiguous()
        enc_mask, dec_mask = (x != 0), (y_inp != 0)

        x, y_inp, y_tar, enc_mask, dec_mask = to_devices(
          (x, y_inp, y_tar, enc_mask, dec_mask), device)

        model.train()
        if aux: freeze_layers(params.frozen_layers, unfreeze=True)
        output = model(input_ids=x, decoder_input_ids=y_inp,
                   labels=y_tar, attention_mask=enc_mask,
                   decoder_attention_mask=dec_mask)
        optimizer.zero_grad()
        output.loss.backward(retain_graph=aux)
        optimizer.step()
        scheduler.step()

        if aux: freeze_layers(params.frozen_layers)
        x_enc = output.encoder_last_hidden_state
        y_enc = model.model.encoder(y_inp, attention_mask=dec_mask)['last_hidden_state']
        x_enc = torch.max(x_enc + -999 * enc_mask.unsqueeze(-1), dim=1)[0]
        y_enc = torch.max(y_enc + -999 * dec_mask.unsqueeze(-1), dim=1)[0]
        aux_loss = F.cosine_embedding_loss(x_enc, y_enc, _target)
        scaled_aux_loss = params.aux_strength * aux_loss
        if aux: scaled_aux_loss.backward()

        accuracy = accuracy_fn(output.logits, y_tar)

        return output.loss.item(), aux_loss.item(), accuracy.item()

    # prepare iterators
    iterators = {direction: iter(loader) for direction, loader in train_dataloaders.items()}
    directions, num_examples = list(num_train_examples.keys()), np.array(list(num_train_examples.values()))
    dir_dist = (num_examples ** params.temp) / ((num_examples ** params.temp).sum())

    #train
    losses, aux_losses, accs = [], [], []
    start_ = time.time()
    for i in range(params.train_steps):

        # sample a direction
        direction = directions[int(np.random.choice(len(num_examples), p=dir_dist))]
        try: # check iterator is not exhausted
            x, y = next(iterators[direction])
        except StopIteration:
            iterators[direction] = iter(train_dataloaders[direction])
            x, y = next(iterators[direction])
        x, y = get_direction(x, y, sample=~params.single_direction)
           
        # train on the direction
        loss, aux_loss, acc = train_step(x, y, aux=params.auxiliary)
        losses.append(loss)
        aux_losses.append(aux_loss)
        accs.append(acc)

        if i % params.verbose == 0:
            print('Batch {} Loss {:.4f} Aux Loss {:.4f} Acc {:.4f} in {:.4f} secs per batch'.format(
                i, np.mean(losses[-params.verbose:]), np.mean(aux_losses[-params.verbose:]),
                np.mean(accs[-params.verbose:]), (time.time() - start_)/(i+1)))
        if params.wandb:
            wandb.log({'train_loss':loss, 'aux_loss':aux_loss, 'train_acc':acc})

    # save results
    if params.save:
        logger.save_model(params.train_steps, model, optimizer, scheduler=scheduler)
    
    train_results = {'loss':[np.mean(losses)], 'aux_loss':[np.mean(aux_losses)], 'accuarcy':[np.mean(accs)]}
    pd.DataFrame(train_results).to_csv(logger.root_path + '/train_results.csv', index=False)

    # evaluate the model
    def evaluate(x, y, y_code, bleu):
        y_inp, y_tar = y[:,:-1].contiguous(), y[:,1:].contiguous()
        enc_mask = (x != 0)
        x, y_inp, y_tar, enc_mask = to_devices(
          (x, y_inp, y_tar, enc_mask), device)
        
        model.eval()
        y_pred = model.generate(input_ids=x, decoder_start_token_id=y_code,
            attention_mask=enc_mask, max_length=params.max_len+1,
            num_beams=params.num_beams, length_penalty=params.length_penalty,
            early_stopping=True)
        bleu(y_pred[:,1:], y_tar)

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
                if params.wandb:
                    wandb.log({'Bleu1':bl1, 'Bleu2':bl2})

        test_results[direction] = [bleu1.get_metric()]
        test_results[alt_direction] = [bleu2.get_metric()]

    # save test_results
    pd.DataFrame(test_results).to_csv(logger.root_path + '/test_results.csv', index=False)

    if params.wandb:
        wandb.finish()


if __name__ == '__main__':

    from common.finetune_arguments import parser
    params = parser.parse_args()
    main(params)

