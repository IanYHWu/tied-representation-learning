'''
Test train.py with some simple parameters.
'''

import torch

from hyperparams.loader import Loader
from train import train

params = {
	'model' : 'base',
	'layers' : 2,
	'heads' : 4,
	'dff' : 64,
	'd_model' : 16,
	'vocab_size' : 20,
	'max_pe' : 20,
	'dropout' : 0.5,
	'warmup_steps' : 1000,
	'lr_scale' : 0.1,
	'epochs' : 10,
	'location' : '.',
	'checkpoint' : False,
	'name' : 'test',
	'custom_model' : False,
	'langs' : ['l1', 'l2']}
params = Loader(params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fake bilingual dataloader for testing
train_dataloader = [
	(torch.randint(0, 20, (32, 10)),
		torch.randint(0, 20, (32, 10))) for _ in range(10)]

train(device, params, train_dataloader, val_dataloader=None, tokenizer=None)

# fake multilingual dataloader and tokenizer
params.langs = ['l1', 'l2', 'l3']
train_dataloader = [
	(torch.randint(4, 20, (32, 10)),
		torch.randint(4, 20, (32, 10)),
		torch.randint(4, 20, (32, 10))) for _ in range(10)]

class Tokenized:
	def __init__(self, x):
		self.ids = [0, x]

class Tokenizer:
	def __init__(self):
		self.vocab =  {'[l1]' : 1, '[l2]' : 2, '[l3]' : 3}
	def encode(self, x):
		return Tokenized(self.vocab[x])
tokenizer = Tokenizer()

train(device, params, train_dataloader, val_dataloader=None, tokenizer=tokenizer)

