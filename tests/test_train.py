import torch
from train import train

vocab_size = 10
batch_size = 2
epochs = 3

transformer_args = {
	'num_layers' : 2,
	'num_heads' : 2,
	'dff' : 64,
	'd_model' : 32,
	'input_vocab_size' : 10,
	'target_vocab_size' : 10,
	'pe_input' : 20,
	'pe_target' : 20,
	'rate' : 0.1}

opt_args = {
	'lr' : 1e-4}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fake dataloader for testing
train_dataloader = [(torch.randint(0, vocab_size, (batch_size, 10)), torch.randint(0, vocab_size, (batch_size, 10))) for _ in range(10)]

train(device, epochs, transformer_args, opt_args, train_dataloader, val_dataloader=None)