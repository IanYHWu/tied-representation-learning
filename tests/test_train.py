'''
Test train.py with some simple parameters.
'''

import torch
from train import train

transformer_args = {
	'num_layers' : 2,
	'num_heads' : 4,
	'dff' : 64,
	'd_model' : 16,
	'input_vocab_size' : 20,
	'target_vocab_size' : 20,
	'pe_input' : 20,
	'pe_target' : 10,
	'rate' : 0.5}

opt_args = {
	'lr' : 1e-3}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fake dataloader for testing
train_dataloader = [
	(torch.randint(0, 20, (32, 10)),
		torch.randint(0, 20, (32, 10))) for _ in range(10)]

train(device, 5, transformer_args, opt_args, train_dataloader, val_dataloader=None)