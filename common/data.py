import numpy as np
import torch
import pytorch_lightning as pl
from datasets import load_dataset
from itertools import combinations
from transformers import MBart50TokenizerFast
from common.preprocess import filter_languages


LANG_CODES = { # mapping ted to mBart lang codes
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


BITEXT_DATASETS = { # dataset used for each language pair
    'en-tr' : 'ted_multi',
    'az-en' : 'ted_multi',
    'az-tr' : 'ted_multi',
    'cs-en' : 'wmt14',
    'en-fr' : 'wmt14',
    'de-en' : 'wmt19',
}


""" Dataset classes """
class TedMulti:
    """ Ted Multilingual Dataset."""

    def __init__(self, langs, batch_size, max_len, tokenizer):
        self.langs = langs 
        self.cols = ['input_ids_' + l for l in langs]
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.dataset = load_dataset('ted_multi')
 
    def load_split(self, split, shuffle=False):
        dataset = self.dataset[split]
        dataset = filter_languages(dataset, self.langs)

        def tokenize_fn(example):
            """apply tokenization"""
            l_tok = []
            for lang in self.langs:
                encoded = self.tokenizer.encode(example[lang], padding='max_length',
                    max_length=self.max_len, truncation=True)
                encoded[0] = self.tokenizer.lang_code_to_id[LANG_CODES[lang]]
                l_tok.append(encoded)
            return {'input_ids_' + l: tok for l, tok in zip(self.langs, l_tok)}

        dataset = dataset.map(tokenize_fn)
        dataset.set_format(type='torch', columns=self.cols)

        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=5)

        return dataloader, len(dataset)


class WMT:
    """ WMT dataset for given year. """

    def __init__(self, langs, batch_size, max_len, tokenizer, name='wmt14'):
        self.langs = sorted(langs)
        self.cols = ['input_ids_' + l for l in langs]
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = tokenizer
        try:
            self.dataset = load_dataset(name, self.langs[0] + '-' + self.langs[1])
        except ValueError:
            self.dataset = load_dataset(name, self.langs[1] + '-' + self.langs[0])

    def load_split(self, split, shuffle=False):
        dataset = self.dataset[split]

        def tokenize_fn(example):
            """apply tokenization"""
            l_tok = []
            for lang in self.langs:
                encoded = self.tokenizer.batch_encode_plus(example[lang], padding='max_length',
                    max_length=self.max_len, truncation=True, return_tensors='pt')['input_ids']
                for seq in encoded:
                    seq[0] = self.tokenizer.lang_code_to_id[LANG_CODES[lang]]
                l_tok.append(encoded)
            return {'input_ids_' + l: tok for l, tok in zip(self.langs, l_tok)}

        def collate_fn(examples):
            lang_keys = examples[0]['translation'].keys()
            examples = {l : [ex['translation'][l] for ex in examples] for l in lang_keys}
            return tokenize_fn(examples)

        dataloader = torch.utils.data.DataLoader(dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size)

        return dataloader, len(dataset)


""" Data loading classes """
class MNMTDataset(torch.utils.data.IterableDataset):
    """ Samples from given datasets using temperature sampling. """

    def __init__(self, datasets, langs, T=1.0, bilingual=False):
        self.datasets = datasets
        self.langs = langs
        self.bilingual = bilingual

        if not self.bilingual:
            self.lang_pairs = list(combinations(self.langs, 2))
            lengths = np.array([len(datasets[l1+'-'+l2]) for l1, l2 in self.lang_pairs])
            self.prob_dist = (lengths ** T) / (lengths ** T).sum()

    def __iter__(self):
        self.iterators = {k:iter(d) for k,d in self.datasets.items()}
        return self

    def __next__(self):
        if self.bilingual:
            l1, l2 = self.langs[0], self.langs[1]
        else:
            k = np.random.choice(len(self.lang_pairs), p=self.prob_dist)
            l1, l2 = self.lang_pairs[k][0], self.lang_pairs[k][1]
        pair = l1 + '-' + l2
        
        try:
            data = next(self.iterators[pair])
        except StopIteration:
            self.iterators[pair] = iter(self.datasets[pair])
            data = next(self.iterators[pair])

        x, y = data['input_ids_' + l1], data['input_ids_' + l2]

        if not(self.bilingual) and (np.random.rand() > 0.5):
            return y, x
        else:
            return x, y

    def next(self):
        return __next__(self)


class MNMTDataModule(pl.LightningDataModule):
    """Training data class for pytorch lightning. """

    def __init__(self, langs, batch_size, max_len, T=1.0):
        super().__init__()
        self.langs = langs
        self.batch_size = batch_size
        self.max_len = max_len
        self.T = T

    def preprare_data(self):
        for l1, l2 in combinations(sorted(self.langs), 2):
            lang_pair = l1 + '-' + l2 
            if BITEXT_DATASETS[lang_pair] == 'ted_multi':
                load_dataset('ted_multi')
            elif BITEXT_DATASETS[lang_pair] == 'wmt14':
                load_dataset('wmt14', lang_pair)
            else:
                raise NotImplementedError
        tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50')

    def setup(self, stage=None):
        self.tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50')
        self.splits = {'train':{}, 'validation':{}, 'test':{}}
        self.train_examples = []
        
        for l1, l2 in combinations(sorted(self.langs), 2):
            lang_pair = l1 + '-' + l2 
            
            if BITEXT_DATASETS[lang_pair] == 'ted_multi':
                dataset = TedMulti([l1, l2], self.batch_size, self.max_len, self.tokenizer)
            elif BITEXT_DATASETS[lang_pair] == 'wmt14':
                dataset = WMT([l1, l2], self.batch_size, self.max_len, self.tokenizer, name='wmt14')
            else:
                raise NotImplementedError
            
            splits = ['train'] if stage == 'fit' else ['validation', 'test']
            for split in splits:
                shuffle = False if split == 'test' else True
                dataset_split, num_examples = dataset.load_split(split, shuffle=shuffle)
                self.splits[split][lang_pair] = dataset_split
                if split=='train': self.train_examples.append(num_examples)

    def train_dataloader(self):
        iterable = MNMTDataset(self.splits['train'], self.langs, T=self.T, bilingual=len(self.langs)==2)
        return torch.utils.data.DataLoader(iterable, batch_size=None)

