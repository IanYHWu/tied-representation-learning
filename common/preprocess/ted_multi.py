import torch
import datasets

from common.preprocess.preprocess_utils import train_tokenizer, pad_sequence, AddTargetTokens
from common.utils import sample_direction, get_direction


def filter_languages(dataset, langs):
    """Extracts translations for given languages
    from each example in the dataset"""

    def filter_fn(example):
        """check all translations exist"""
        return all([l in example['translations']['language'] for l in langs])

    def map_fn(example):
        """get the translations"""
        idx = [example['translations']['language'].index(l) for l in langs]
        return {l: example['translations']['translation'][id] for l, id in zip(langs, idx)}

    dataset = dataset.filter(filter_fn).map(map_fn, remove_columns=['talk_name', 'translations'])

    return dataset


def preprocess(dataset, langs, batch_size=32, tokenizer=None, vocab_size=None, max_len=None,
    multi=False, distributed=False, world_size=None, rank=None):
    """
    Applies full preprocessing to dataset: filtering, tokenization,
    padding and conversion to torch dataloader.

    dataset : HuggingFace Ted-Multi Dataset
    langs : list of languages
    batch_size : int - batch size for dataloader
    tokenizers : list of pretrained tokenizers or None (if None tokenizers will be trained).
    vocab_size : int - vocab size for tokenization (only needed if tokenziers is None)
    max_len : int - maximum allowed length of sequences.
    multi : bool - wether to do multilingual preprocessing.
    distributed : bool - wether to set up a distributed dataset.
    world_size : int - number of processes * gpus (only needed for distributed).
    rank : int - rank of the process (only needed for distributed).

    Returns:

    """

    # filtering
    dataset = filter_languages(dataset, langs)

    # tokenization
    if multi:
        if tokenizer is None:
            tokenizer = train_tokenizer(langs, dataset, vocab_size)

        def tokenize_fn(example):
            """apply tokenization"""
            l_tok = [tokenizer.encode(example[l]) for l in langs]
            return {'input_ids_' + l: tok.ids for l, tok in zip(langs, l_tok)}
    else:
        if tokenizer is None:
            tokenizer = [train_tokenizer([lang], dataset, vocab_size) for lang in langs]

        def tokenize_fn(example):
            """apply tokenization"""
            l_tok = [tok.encode(example[l]) for tok, l in zip(tokenizer, langs)]
            return {'input_ids_' + l: tok.ids for l, tok in zip(langs, l_tok)}
    
    dataset = dataset.map(tokenize_fn)

    # padding and convert to torch
    cols = ['input_ids_' + l for l in langs]
    dataset.set_format(type='torch', columns=cols)

    def pad_seqs(examples):
        """Apply padding"""
        ex_langs = list(zip(*[tuple(ex[col] for col in cols) for ex in examples]))
        ex_langs = tuple(pad_sequence(x, batch_first=True) for x in ex_langs)
        return ex_langs

    print("Dataset Size: {}".format(len(dataset[langs[0]])))

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


class TedMultiDataLoader:
    """
    Creats a dataloader for ted multi dataset for a given mode.

    Mode behaviours:
        bilingual: always returns x, y as source, target.

        multi: during tarining samples a direction and returns
        source, target, source language and target language. If
        testing then returns a dict of all generated translation
        pairs.

        pivot: returns the directions required for pivot training.
    """

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


def load_ted_multi(langs, vocab_size, batch_size=32, mode='bilingual',
    tokenizer=None, pivot_pair_ind=None, max_len=None, path=None, distributed=False,
    world_size=None, rank=None, excluded=None):
    """Load and preprocess the ted-multilingual dataset.
    langs : list of language ids
    batch_size : batch_size for the dataloaders (per gpu).
    vocab_size : size of the vocab(s) of tokenizer(s)
    dataset_name : string name for the huggingface dataset
    tokenizer : tokenizer or list of tokenizers. If None tokenizer is trained.
    multi : bool = True, wether to use a shared tokenizer for all languages
    path : str = None, if given the location where the tokenizer will be saved.
    distributed : bool - wether to set up a distributed training dataset.
    world_size : int - number of processes * gpus (only needed for distributed).
    rank : int - rank of the process (only needed for distributed)
    excluded : list - list of tuples of excluded translation directions.

    Returns: preprocessed dataloaders for train, val and test splits and
    the trained tokenizer(s).
    
    Note on distributed training: with distributed training the batch_size
    given should be the batch size per gpu. So each step there will be
    batch_size * world_size examples processed. Also note that the val and
    test datasets are not distributed and are only processed on rank 0.
    """

    dataset = datasets.load_dataset('ted_multi')

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    save_tokenizer = True if tokenizer is None else False
    multi = True if mode == 'multi' else False

    train_dataloader, tokenizer = preprocess(train_dataset, langs, batch_size=batch_size, tokenizer=tokenizer,
        vocab_size=vocab_size, max_len=max_len, multi=multi, distributed=distributed, world_size=world_size, rank=rank)

    val_dataloader, _ = preprocess(val_dataset, langs, batch_size=batch_size, tokenizer=tokenizer, max_len=max_len,
        multi=multi, distributed=False)

    test_dataloader, _ = preprocess(test_dataset, langs, batch_size=batch_size, tokenizer=tokenizer, max_len=max_len,
        multi=multi, distributed=False)
    
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


if __name__ == "__main__":

    _dataloader = [(0,1), (0,2), (3,4), (4,5)]
    dataloader = TedMultiDataLoader('bilingual', ['en', 'es'], _dataloader)
    for data in dataloader:
        print(data)