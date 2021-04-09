"""
Pipeline for loading the TED-Multilingual dataset via HuggingFace,
training BPE tokenizers for given languages and creating a torch dataloader.

If doing multi-lingual translation then a single, shared tokenizer is trained 
however for bilingual translation individual tokenizers are used.
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

	# pre tokenizer with whitespace
	tokenizer.pre_tokenizer = Whitespace()

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
		tokenizers = [train_tokenizer([lang], dataset, vocab_size) for lang in langs]

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

def preprocess_multi(dataset, langs, batch_size=32, tokenizer=None, vocab_size=None):
	"""Applies full preprocessing to dataset: filtering, tokenization,
	padding and conversion to torch dataloader.
	dataset : HuggingFace Ted-Multi Dataset
	langs : list of languages
	batch_size : int - batch size for dataloader
	tokenizers : pretrained tokenizers or None
	(if None tokenizer will be trained).'
	vocab_size : int - vocab size for tokenization
	(only needed if tokenziers is None)"""

	# filtering
	dataset = filter_languages(dataset, langs)

	# tokenization
	if tokenizer is None:
		tokenizer = train_tokenizer(langs, dataset, vocab_size)

	def tokenize_fn(example):
		"""apply tokenization"""
		l_tok = [tokenizer.encode(example[l]) for l in langs]
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

	return dataloader, tokenizer

def load_and_preprocess(langs, batch_size, vocab_size, dataset_name, multi=True):
	"""Load and preprocess the data.
	langs : list of language ids
	batch_size : batch_size for the dataloaders.
	vocab_size : size of the vocab(s) of tokenizer(s)
	dataset_name : string name for the huggingface dataset
	multi : bool = True, wether to use a shared tokenizer for all languages.

	Returns: preprocessed dataloaders for train, val and test splits and
	the trained tokenizer(s)."""

	dataset = datasets.load_dataset(dataset_name)

	train_dataset = dataset['train']
	val_dataset = dataset['validation']
	test_dataset = dataset['test']

	if multi:
		train_dataloader, tokenizer = preprocess_multi(train_dataset, langs,
			batch_size=batch_size,
			tokenizer=None,
			vocab_size=vocab_size)
		val_dataloader, _ = preprocess_multi(val_dataset, langs, batch_size=batch_size, tokenizer=tokenizer)
		test_dataloader, _ = preprocess_multi(test_dataset, langs, batch_size=batch_size, tokenizer=tokenizer)
		return train_dataloader, val_dataloader, test_dataloader, tokenizer
	else:
		train_dataloader, tokenizers = preprocess(train_dataset, langs,
			batch_size=batch_size,
			tokenizers=None,
			vocab_size=vocab_size)
		val_dataloader, _ = preprocess(val_dataset, langs, batch_size=batch_size, tokenizers=tokenizers)
		test_dataloader, _ = preprocess(test_dataset, langs, batch_size=batch_size, tokenizers=tokenizers)
		return train_dataloader, val_dataloader, test_dataloader, tokenizers

def _detokenize(x, tokenizer, as_lists = True):
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

def detokenize(x, tokenizer, as_lists = True):
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
			return [_detokenize(x_, tok, as_lists = as_lists) for x_, tok in zip(x, tokenizer)]
		else:
			return [_detokenize(x_, tokenizer, as_lists = as_lists) for x_ in x]
	else:
		return _detokenize(x, tokenizer, as_lists = as_lists)

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
		lang : tag of desired target language

		Returns: batch of tensors (batch, seq_len + 1)
		where the dimension x[:,0] now contains the token
		for the target language token.
		"""
		token = self.tokenizer.encode('[' + lang + ']').ids[1]
		tokens = torch.ones(x.shape[0], 1, dtype = torch.long) * token
		return torch.cat([tokens, x], dim = 1)


if __name__ == "__main__":

	# !!!! very large download !!!
	train, val, test, tokenizers = load_and_preprocess(['en', 'fr', 'de'], 32, 10000, "ted_multi", multi = True)