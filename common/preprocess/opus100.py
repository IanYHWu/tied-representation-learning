import torch 
import datasets
from datasets import concatenate_datasets

from common.preprocess.preprocess_utils import train_tokenizer, pad_sequence, AddTargetTokens
from common.utils import sample_direction, get_direction


def get_opus_pairs(langs, excluded=None):
    """ opus100 requires a language pair as 'lang1-lang2'
    in alphabetical order so need to prepare pairs from a
    list of given languages. """

    # get language pairs for loading and remove excluded
    assert 'en' in langs
    langs = sorted(langs)
    langs_pre, langs_post = langs[:langs.index('en')], langs[langs.index('en')+1:]
    pairs = [lang + '-en' for lang in langs_pre] + ['en-' + lang for lang in langs_post]

    # remove excluded
    if excluded is not None:
        excluded = [sorted(pair)[0] + '-' + sorted(pair)[1] for pair in excluded]
        pairs = [pair for pair in pairs if pair not in excluded]

    return pairs


def map_fn(example):
    """ Change to 4 columns: lang1, text1, lang2, text2."""
    example = example['translation']
    keys = list(example.keys())
    return {'lang0' : keys[0], 'text0' : example[keys[0]],
            'lang1' : keys[1], 'text1' : example[keys[1]]}


def load_from_pairs(pairs):
    """ construct an Opus100 dataset from translation pairs. """

    # load and map each pair to common columns
    all_ = []
    for pair in pairs:
        dataset = datasets.load_dataset('opus100', pair)
        for k, v in dataset.items():
            dataset[k] = v.map(map_fn, remove_columns=['translation'])
        all_.append(dataset)

    # concatenate each split
    all_datasets = {}
    for split in ['train', 'test', 'validation']:
        all_datasets[split] = concatenate_datasets([dataset_[split] for dataset_ in all_])

    return all_datasets


def preprocess(dataset, langs, batch_size=32, tokenizer=None, vocab_size=None, max_len=None,
    multi=False, distributed=False, world_size=None, rank=None):
    """ preprocess an opus100 dataset by tokenization and setting a dataloader. """

    lang_columns = ['text0', 'text1'] # columns where the translations are found

    # tokenization
    if multi:
        if tokenizer is None:
            tokenizer = train_tokenizer(langs, dataset, vocab_size, lang_columns=lang_columns)

        def tokenize_fn(example):
            """apply tokenization"""
            l_tok = [tokenizer.encode(example[l]) for l in lang_columns]
            for l, tok in zip(lang_columns, l_tok):
                example['input_ids_' + l] = tok.ids
            return example
    else:
        if tokenizer is None:
            tokenizer = [train_tokenizer(
                [lang], dataset, vocab_size, lang_columns=[l_col]
                ) for lang, l_col in zip(langs, lang_columns)]

        def tokenize_fn(example):
            """apply tokenization"""
            l_tok = [tok.encode(example[l]) for tok, l in zip(tokenizer, langs)]
            for l, tok in zip(lang_columns, l_tok):
                example['input_ids_' + l] = tok.ids 
            return example
    
    dataset = dataset.map(tokenize_fn, remove_columns=lang_columns)

    # padding and convert to torch
    cols = ['input_ids_text0', 'input_ids_text1'] # input_ids, lang
    dataset.set_format(type='torch', columns=cols, output_all_columns=True)

    def pad_seqs(examples):
        """Apply padding"""
        columns = cols + ['lang0', 'lang1']
        x0, x1, lang0, lang1 = list(zip(*[[ex[col] for col in columns] for ex in examples]))
        x0 = U.pad_sequence(x0, batch_first=True, max_len=max_len)
        x1 = U.pad_sequence(x1, batch_first=True, max_len=max_len)
        return x0, x1, lang0, lang1

    print("Dataset Size: {}".format(len(dataset)))

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
            num_replicas=world_size,
            rank=rank)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 collate_fn=pad_seqs,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 collate_fn=pad_seqs,
                                                 shuffle=True)

    return dataloader, tokenizer


class Opus100DataLoader:
    """
    Creates a dataloader for opus dataset for a given mode.

    Mode behaviours:
        bilingual: always returns x, y as source, target.

        multi: during training samples a direction and returns
        source, target, source language and target language. If
        testing then returns a dict of all generated translation
        pairs.

        pivot: returns the directions required for pivot training.
    """

    def __init__(self, mode, langs, dataloader, tokenizer=None, pivot_pair_ind=None, test=False):
        self.mode = mode 
        self.langs = langs 
        self.dataloader = dataloader
        self.test = test
        self.tokenizer = tokenizer

        if self.mode == 'pivot':
            assert self.pivot_pair_ind is not None
            self.pivot_pair0, self.pivot_pair1 = pivot_pair_ind

        elif self.mode == 'multi':
            assert self.tokenizer is not None
            self.add_targets = AddTargetTokens(langs, tokenizer)

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        x, y, x_lang, y_lang = next(self.iterator)

        if self.mode == 'bilingual':
            return x, y
        
        elif self.mode == 'multi':
            return self.get_multi(data)
       
        elif self.mode == 'pivot':
            raise NotImplementedError
        
        else:
            raise NotImplementedError

    def get_multi(self, data):
        if self.test:
            raise NotImplementedError
        else:
            # combine both translation directions add add targets
            (x, y), (x_lang, y_lang) = sample_direction(data, self.langs, excluded=self.excluded)
            x_comb, y_comb = torch.cat([x, y], dim=0), torch.cat([y, x], dim=0)
            x_comb = self.add_targets(x_comb, y_lang + x_lang)
            return x_comb, y_comb


def load_opus100(langs, vocab_size, batch_size=32, mode='bilingual', tokenizer=None,
    pivot_pair_ind=None, max_len=None, path=None, distributed=False,
    world_size=None, rank=None, excluded=None, test_only=False):
    
    # load datasets and concatenate
    pairs = get_opus_pairs(langs, excluded=excluded)    
    all_datasets = load_from_pairs(pairs)
    train_dataset = all_datasets['train']
    val_dataset = all_datasets['validation']
    test_dataset = all_datasets['test']

    multi = True if mode == 'multi' else False

    if mode == 'pivot':
        raise NotImplementedError

    if not test_only:

        save_tokenizer = True if tokenizer is None else False
        
        if multi:
            train_batch_size = batch_size // 2
        else:
            train_batch_size = batch_size

        # tokenize and dataloaders
        train_dataloader, tokenizer = preprocess(train_dataset, langs, batch_size=train_batch_size,
            tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len, multi=multi,
            distributed=distributed, world_size=world_size, rank=rank)
        val_dataloader, _ = preprocess(val_dataset, langs, batch_size=batch_size,
            tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len, multi=multi,
            distributed=distributed, world_size=world_size, rank=rank)
        test_dataloader, _ = preprocess(test_dataset, langs, batch_size=batch_size,
            tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len, multi=multi,
            distributed=distributed, world_size=world_size, rank=rank)

        # save tokenizers if trained
        if (path is not None) and save_tokenizer:
            if isinstance(tokenizer, list):
                for tok, lang in zip(tokenizer, langs):
                    tok.save(path + '/' + lang + '_tokenizer.json')
            else:
                tokenizer.save(path + '/multi_tokenizer.json')

        # create dataloaders
        train_dataloader = Opus100DataLoader(mode, langs, train_dataloader,
            tokenizer=tokenizer, pivot_pair_ind=pivot_pair_ind, test=False)
        val_dataloader = Opus100DataLoader(mode, langs, val_dataloader,
            tokenizer=tokenizer, pivot_pair_ind=pivot_pair_ind, test=False)
        test_dataloader = Opus100DataLoader(mode, langs, test_dataloader,
            tokenizer=tokenizer, pivot_pair_ind=pivot_pair_ind, test=True)

        return train_dataloader, val_dataloader, test_dataloader, tokenizer

    else:

        if tokenizer is None:
            print('Tokenizer required for test only.')
            raise ValueError

        test_dataloader, _ = preprocess(test_dataset, langs, batch_size=batch_size,
            tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len, multi=multi,
            distributed=distributed, world_size=world_size, rank=rank)
        test_dataloader = Opus100DataLoader(mode, langs, test_dataloader,
            tokenizer=tokenizer, pivot_pair_ind=pivot_pair_ind, test=True)

        return test_dataloader



