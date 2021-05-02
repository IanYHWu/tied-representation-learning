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
    pairs = [lang + '-en' for lang in langs_pre] + ['en-' + lang for lang in langs_pre]

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
    all = []
    for pair in pairs:
    dataset = datasets.load_dataset('opus100', pair)
    for k, v in dataset.values():
        dataset[k] = v.map(map_fn, remove_columns=['translation'])
    all.append(dataset)

    # concatenate each split
    all_datasets = {}
    for split in ['train', 'test', 'validation']:
        all_datasets[split] = concatenate_datasets([dataset_[split] for dataset_ in all])

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
                ) for lang, l_col in zip(lang, lang_columns)]

        def tokenize_fn(example):
            """apply tokenization"""
            l_tok = [tok.encode(example[l]) for tok, l in zip(tokenizer, langs)]
            for l, tok in zip(lang_columns, l_tok):
                example['input_ids_' + l] = tok.ids 
            return example
    
    dataset = dataset.map(tokenize_fn, remove_columns=lang_columns)

    # padding and convert to torch
    cols = dataset.column_names # input_ids, lang
    dataset.set_format(type='torch', columns=cols)

    def pad_seqs(examples):
        """Apply padding"""
        ex_langs = list(zip(*[tuple(ex[col] for col in lang_columns) for ex in examples]))
        ex_langs = tuple(pad_sequence(x, batch_first=True, max_len=max_len) for x in ex_langs)
        return ex_langs

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
                                                 collate_fn=pad_seqs)

    return dataloader, tokenizer


class Opus100DataLoader:
    """
    Creats a dataloader for opus dataset for a given mode.

    Mode behaviours:
        bilingual: always returns x, y as source, target.

        multi: during tarining samples a direction and returns
        source, target, source language and target language. If
        testing then returns a dict of all generated translation
        pairs.

        pivot: returns the directions required for pivot training.
    """







    ############## incomplete from here #################

    def __init__(self, mode, langs, dataloader, tokenizer=None, pivot_pair_ind=None, test=False, excluded=None):
        self.mode = mode 
        self.langs = langs 
        self.dataloader = dataloader
        self.test = test 
        self.excluded = excluded
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
        data = next(self.iterator)

        if self.mode == 'bilingual':
            return data
        
        elif self.mode == 'multi':
            return self.get_multi(data)
       
        elif self.mode == 'pivot':
            x, y = get_direction(data, self.pivot_pair0, self.pivot_pair1, excluded=self.excluded)
            return x, y
        
        else:
            raise NotImplementedError

    def get_multi(self, data):
        if self.test:
            # gets all directions and adds target tokens
            data = get_directions(data, self.langs, excluded=self.excluded)
            for direction, (x, y, y_lang) in data.items():
                x = self.add_targets(x, y_lang)
                data[direction] = (x, y, y_lang)
            return data
        else:
            # sample a tranlsation direction and add target tokens
            (x, y), (x_lang, y_lang) = sample_direction(data, self.langs, excluded=self.excluded)
            x = self.add_targets(x, y_lang)
            return x, y






            ############## incomplete from here #################

def load_opus100(langs, vocab_size, batch_size=32, mode='bilingual', tokenizer=None,
    pivot_pair_ind=None, max_len=None, path=None, distributed=False,
    world_size=None, rank=None, excluded=None):
    
    # load datasets and concatenate
    pairs = get_opus_pairs(langs, excluded=excluded)    
    all_datasets = load_from_pairs(pairs)
    train_dataset = all_datasets['train']
    val_dataset = all_datasets['train']
    test_dataset = all_datasets['train']

    save_tokenizer = True if tokenizer is None else False
    multi = True if mode == 'multi' else False

    # tokenize and dataloaders
    train_dataloader, tokenizer = preprocess(train_dataset, langs, batch_size=batch_size,
        tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len, multi=multi,
        distributed=distributed, world_size=world_size, rank=rank)
    val_dataloader, _ = preprocess(train_dataset, langs, batch_size=batch_size,
        tokenizer=tokenizer, vocab_size=vocab_size, max_len=max_len, multi=multi,
        distributed=distributed, world_size=world_size, rank=rank)
    test_dataloader, _ = preprocess(train_dataset, langs, batch_size=batch_size,
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
    train_dataloader = TedMultiDataLoader(mode, langs, train_dataloader,
        tokenizer=tokenizer, pivot_pair_ind=pivot_pair_ind, test=False, excluded=excluded)
    val_dataloader = TedMultiDataLoader(mode, langs, val_dataloader,
        tokenizer=tokenizer, pivot_pair_ind=pivot_pair_ind, test=False, excluded=excluded)
    test_dataloader = TedMultiDataLoader(mode, langs, test_dataloader,
        tokenizer=tokenizer, pivot_pair_ind=pivot_pair_ind, test=True, excluded=excluded)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer

