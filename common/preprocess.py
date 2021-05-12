"""
Pipeline for loading translation dataset via HuggingFace, training BPE
tokenizers for given languages and creating a torch dataloader.

If doing multi-lingual translation then a single, shared tokenizer is trained 
however for bilingual translation individual tokenizers are used.

For a dataset to be loadable it needs a filter_languages function which will
extract the required translations from each example.
"""

import torch
import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from tokenizers import pre_tokenizers
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing
from common.utils import remove_after_stop


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


class _MultilingualIterator:
    """ Simple iterator class to combine languages."""

    def __init__(self, dataset, langs):
        self.dataset = dataset
        self.langs = langs

    def __iter__(self):
        self.iterator = iter(self.dataset)
        return self

    def __next__(self):
        x = next(self.iterator)
        return ''.join([x[lang] + ' ' for lang in self.langs])


def train_tokenizer(langs, dataset, vocab_size):
    """Train a tokenizer on given list of languages.
    Reserves a special token for each language which is
    [LANG] where LANG is the language tag. These are assigned
    to tokens 5, 6, ..., len(langs) + 4.
    """

    # Byte-pair encoding
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))

    # trainer
    lang_tokens = ['[' + lang + ']' for lang in langs]
    special_tokens = ['[MASK]', '[CLS]', '[SEP]', '[PAD]', '[UNK]'] + lang_tokens
    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=vocab_size)

    # normalise and pre tokenize
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    # create iterator and train
    iterator = _MultilingualIterator(dataset, langs)
    tokenizer.train_from_iterator(iterator, trainer)

    # post process start/end tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ], )
    return tokenizer


def pad_sequence(sequences, batch_first=False, padding_value=0.0, max_len = None):
    """ Same as torch.nn.utils.rnn.pad_sequence but adds the
    option of padding sequences to a fixed length."""
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = min([tensor.size(0), max_len])
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor[:length]
        else:
            out_tensor[:length, i, ...] = tensor[:length]

    return out_tensor


def preprocess(dataset, langs, batch_size=32, tokenizer=None, vocab_size=None, max_len=None,
    multi=False, distributed=False, world_size=None, rank=None, load=True):
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


def load_and_preprocess(langs, batch_size, vocab_size, dataset_name,
    tokenizer=None, multi=True, max_len=None, path=None, distributed=False,
    world_size=None, rank=None):
    """Load and preprocess the data.
    langs : list of language ids
    batch_size : batch_size for the dataloaders (per gpu).
    vocab_size : size of the vocab(s) of tokenizer(s)
    dataset_name : string name for the huggingface dataset
    tokenizer : tokenizer or list of tokenizers. If None tokenizer is trained.
    multi : bool = True, wether to use a shared tokenizer for all languages
    path : str = None, if given the location where the tokenizer will be saved.
    distributed : bool - wether to set up a distributed training dataset.
    world_size : int - number of processes * gpus (only needed for distributed).
    rank : int - rank of the process (only needed for distributed).

    Returns: preprocessed dataloaders for train, val and test splits and
    the trained tokenizer(s).
    
    Note on distributed training: with distributed training the batch_size
    given should be the batch size per gpu. So each step there will be
    batch_size * world_size examples processed. Also note that the val and
    test datasets are not distributed and are only processed on rank 0.
    """

    dataset = datasets.load_dataset(dataset_name)

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    save_tokenizer = True if tokenizer is None else False

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

    return train_dataloader, val_dataloader, test_dataloader, tokenizer


def pivot_preprocess(dataset, langs, tokenizer_1, tokenizer_2, batch_size=32):

    # filtering
    dataset = filter_languages(dataset, langs)
    dataset_1 = dataset.remove_columns(langs[2])
    dataset_2 = dataset.remove_columns(langs[0])
    langs_1 = [langs[0], langs[1]]
    langs_2 = [langs[1], langs[2]]

    def tokenize_fn_1(example):
        """apply tokenization"""
        l_tok = [tok.encode(example[l]) for tok, l in zip(tokenizer_1, langs_1)]
        return {'input_ids_' + l: tok.ids for l, tok in zip(langs_1, l_tok)}

    def tokenize_fn_2(example):
        """apply tokenization"""
        l_tok = [tok.encode(example[l]) for tok, l in zip(tokenizer_2, langs_2)]
        return {'input_ids_' + l: tok.ids for l, tok in zip(langs_2, l_tok)}

    dataset_1 = dataset_1.map(tokenize_fn_1)
    dataset_2 = dataset_2.map(tokenize_fn_2)

    # padding and convert to torch
    cols_1 = ['input_ids_' + l for l in langs_1]
    cols_2 = ['input_ids_' + l for l in langs_2]
    dataset_1.set_format(type='torch', columns=cols_1)
    dataset_2.set_format(type='torch', columns=cols_2)

    def pad_seqs_1(examples):
        """Apply padding"""
        ex_langs = list(zip(*[tuple(ex[col] for col in cols_1) for ex in examples]))
        ex_langs = tuple(pad_sequence(x, batch_first=True) for x in ex_langs)
        return ex_langs

    def pad_seqs_2(examples):
        """Apply padding"""
        ex_langs = list(zip(*[tuple(ex[col] for col in cols_2) for ex in examples]))
        ex_langs = tuple(pad_sequence(x, batch_first=True) for x in ex_langs)
        return ex_langs

    print("Dataset Size: {}".format(len(dataset_1[langs[0]])))

    dataloader_1 = torch.utils.data.DataLoader(dataset_1,
                                             batch_size=batch_size,
                                             collate_fn=pad_seqs_1)
    dataloader_2 = torch.utils.data.DataLoader(dataset_2,
                                               batch_size=batch_size,
                                               collate_fn=pad_seqs_2)

    return dataloader_1, dataloader_2


def pivot_load_and_preprocess(langs, batch_size, dataset_name, tokenizer_1, tokenizer_2):

    dataset = datasets.load_dataset(dataset_name)

    test_dataset = dataset['test']

    test_dataloader_1, test_dataloader_2 = \
        pivot_preprocess(test_dataset, langs, tokenizer_1, tokenizer_2, batch_size=32)

    return test_dataloader_1, test_dataloader_2


def _detokenize(x, tokenizer, as_lists=True):
    """ Detokenize a batch of given tensors."""
    batch = True
    if x.ndim == 1:
        batch = False
        x = x.unsqueeze(0)

    x = x.detach().cpu().tolist()
    x = tokenizer.decode_batch(x)

    if as_lists:
        x = [s.split() for s in x]

    if batch:
        return x
    else:
        return x[0]


def detokenize(x, tokenizer, as_lists=True):
    """
    Detokenize a given batch of sequences of tokens (or
    a list of batches of sequences of tokens).
    x : int torch.tensors (or list of) - shape (batch, seq_len)
    tokenizer : tokenizers.Tokenizer
    as_list : bool - return as list or string

    Returns:
    if as_list is True then returns a list of the detokenized sequences
    as a list of tokens. If as_list if Fales then the list consists of strings.
    If a list of batches is passed then the return is a list of the detokenized
    batches.
    """
    if isinstance(x, list):
        if isinstance(tokenizer, list):
            assert len(x) == len(tokenizer)
            return [_detokenize(x_, tok, as_lists=as_lists) for x_, tok in zip(x, tokenizer)]
        else:
            return [_detokenize(x_, tokenizer, as_lists=as_lists) for x_ in x]
    else:
        return _detokenize(x, tokenizer, as_lists=as_lists)


def _tokenize(x, tokenizer):
    """ Tokenize a list of lists."""
    x = remove_after_stop(x)
    x = " ".join(x)
    tok = tokenizer.encode(x).ids
    return tok


def tokenize(x, tokenizer):
    """
    Tokenize a given batch of sequences of words (or
    a list of batches of sequences of words).
    x : int torch.tensors (or list of) - shape (batch, seq_len)
    tokenizer : tokenizers.Tokenizer
    as_list : bool - return as list or string

    Returns:
    if as_list is True then returns a list of the tokenized sequences
    as a list of tokens. If as_list if False then the list consists of strings.
    If a list of batches is passed then the return is a list of the tokenized
    batches.
    """
    detok_list = [_tokenize(x_, tokenizer) for x_ in x]
    max_len = 0
    for i in detok_list:
        if len(i) > max_len:
            max_len = len(i)
    tensor_list = []
    for i in detok_list:
        if len(i) < max_len:
            pad_len = max_len - len(i)
            padding = [0 for i in range(0, pad_len)]
            i += padding
        tensor_list.append(torch.tensor(i))
    detok_tensor = torch.stack(tensor_list)
    return detok_tensor


class AddTargetTokens:
    """
    Class to add target language tokens to batches of tensors.
    Takes a language tag and adds the token for the given language
    to a batch of tokens.
    """

    def __init__(self, langs, tokenizer):
        """
        langs : list of language ids.
        tokenizer : shared tokenizer between languages.
        """
        self.langs = langs
        self.tokenizer = tokenizer

    def __call__(self, x, lang):
        """
        x : batch of tensors (batch, seq_len)
        lang : tag of desired target language or list of target_languages

        Returns: batch of tensors (batch, seq_len + 1)
        where the dimension x[:,0] now contains the token
        for the target language token.
        """
        if isinstance(lang, list):
            tokens = [x_.ids[1] for x_ in self.tokenizer.encode_batch(['[' + l_ + ']' for l_ in lang])]
            tokens = torch.LongTensor(tokens).unsqueeze(-1)
        else:
            token = self.tokenizer.encode('[' + lang + ']').ids[1]
            tokens = torch.ones(x.shape[0], 1, dtype=torch.long) * token
        return torch.cat([tokens, x], dim=1)


if __name__ == "__main__":
    # !!!! very large download !!!
    # train, val, test, tokenizers = load_and_preprocess(['en', 'fr', 'de'], 32, 10000, "ted_multi", multi=True)
    train, val, test, tokenizers = load_and_preprocess(['en', 'fr'], 32, 10000, "ted_multi", multi=False)

    for i, data in enumerate(val):
        x, y = data
        print(detokenize(x, tokenizers[0]))
        break

