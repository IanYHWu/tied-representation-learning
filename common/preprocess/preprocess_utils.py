import torch
import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

class _MultilingualIterator:
    """ Simple iterator class to combine languages.
    langs : the column identifier for each language in the dataset.
    e.g for Ted this is the langauge. for Ous it is lang0, lang1."""

    def __init__(self, dataset, langs):
        self.dataset = dataset
        self.langs = langs

    def __iter__(self):
        self.iterator = iter(self.dataset)
        return self

    def __next__(self):
        x = next(self.iterator)
        return ''.join([x[lang] + ' ' for lang in self.langs])


def train_tokenizer(langs, dataset, vocab_size, lang_columns=None):
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

    # pre tokenizer with whitespace
    tokenizer.pre_tokenizer = Whitespace()

    # create iterator and train
    if lang_columns is None: lang_columns = langs
    iterator = _MultilingualIterator(dataset, lang_columns)
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
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


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

        