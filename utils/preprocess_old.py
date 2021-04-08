"""
Pipeline for loading the TED-Multilingual dataset via HuggingFace,
training a BPE tokenizer and creating a torch dataloader.
"""

import torch
import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


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


def train_tokenizer(lang, dataset, vocab_size):
    # Byte-pair encoding
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))

    # trainer
    trainer = BpeTrainer(
        special_tokens=['[MASK]', '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
        vocab_size=vocab_size)

    # pre tokenizer with whitespace
    tokenizer.pre_tokenizer = Whitespace()

    # train
    tokenizer.train_from_iterator(dataset[lang], trainer)

    # post process start/end tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ], )
    return tokenizer


def preprocess(dataset, langs, batch_size=32, tokenizers=None, vocab_size=None):
    """Applies full preprocessing to dataset: filtering, tokenization,
    padding and conversion to torch dataloader.
    dataset : HuggingFace Ted-Multi Dataset
    langs : list of languages
    batch_size : int - batch size for dataloader
    tokenizers : list of pretrained tokenizers or None
    (if None tokenizers will be trained).'
    vocab_size : int - vocab size for tokenization
    (only needed if tokenziers is None)"""

    # filtering
    dataset = filter_languages(dataset, langs)

    # tokenization
    if tokenizers is None:
        tokenizers = [train_tokenizer(lang, dataset, vocab_size) for lang in langs]

    def tokenize_fn(example):
        """apply tokenization"""
        l_tok = [tokenizer.encode(example[l]) for tokenizer, l in zip(tokenizers, langs)]
        return {'input_ids_' + l: tok.ids for l, tok in zip(langs, l_tok)}

        # padding and convert to torch

    dataset = dataset.map(tokenize_fn)

    # padding and convert to torch
    cols = ['input_ids_' + l for l in langs]
    dataset.set_format(type='torch', columns=cols)

    def pad_seqs(examples):
        """Apply padding"""
        ex_langs = list(zip(*[tuple(ex[col] for col in cols) for ex in examples]))
        ex_langs = tuple(torch.nn.utils.rnn.pad_sequence(x, batch_first=True) for x in ex_langs)
        return ex_langs

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             collate_fn=pad_seqs)

    return dataloader, tokenizers


def load_and_preprocess(langs, batch_size, vocab_size, dataset_name):
    """Load and preprocess the data"""
    dataset = datasets.load_dataset(dataset_name)

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    train_dataloader, tokenizers = preprocess(train_dataset, langs,
                                              batch_size=batch_size,
                                              tokenizers=None,
                                              vocab_size=vocab_size)

    val_dataloader = preprocess(val_dataset, langs, batch_size=batch_size, tokenizers=tokenizers)
    test_dataloader = preprocess(test_dataset, langs, batch_size=batch_size, tokenizers=tokenizers)

    return train_dataloader, val_dataloader, test_dataloader, tokenizers

def detokenize(x, tokenizer, as_lists = True):
    """
    Detokenize a given batch of sequences of tokens.
    x : int torch.tensors - shape (batch, seq_len)
    tokenizer : tokenizers.Tokenizer
    as_list : bool - return as list or string
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)

    x = x.detach().cpu().tolist()
    x = tokenizer.decode_batch(x)

    if as_lists:
        return [s.split() for s in x]
    else:
        return x

if __name__ == "__main__":

    # !!!! very large download !!!
    train, val, test, tokenizers = load_and_preprocess(['en', 'fr'], 32, 2000, "ted_multi")
