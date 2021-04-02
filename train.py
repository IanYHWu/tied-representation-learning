"""
Training Loop for MNMT
"""

import numpy as np
import torch
import models.base_transformer as base_transformer
import utils.preprocess as preprocess
import utils.metrics as metrics
import time

def to_devices(tensors, device):
	return (tensor.to(device) for tensor in tensors)

def train(device, epochs, model_kwargs, opt_args, train_dataloader, val_dataloader=None):
	"""Training Loop"""
	model = base_transformer.Transformer(**model_kwargs).to(device)
	optimizer = torch.optim.Adam(model.parameters(), **opt_args)
	criterion = torch.nn.CrossEntropyLoss(reduction = 'none')

	def loss_fn(y_pred, y_true):
		_mask = torch.logical_not(y_true == 0).float()
		_loss = criterion(y_pred, y_true)
		return (_loss * _mask).sum() / _mask.sum()
	
	def accuracy_fn(y_pred, y_true):
		_mask = torch.logical_not(y_true == 0).float()
		_acc = (torch.argmax(y_pred, axis=-1) == y_true)
		return (_acc * _mask).sum() / _mask.sum()

	# define train and val steps
	def train_step(x, y):
	
		# get masks and targets
		y_inp, y_tar = y[:,:-1], y[:,1:]
		enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)
		
		# devices
		x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask = to_devices(
			(x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask),
			device)

		# forward
		model.train()
		y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)
		loss = loss_fn(y_pred.permute(0, 2, 1), y_tar)
		
		# backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# metrics
		batch_loss = loss.cpu().item()
		batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()

		return batch_loss, batch_acc

	def val_step(x, y):
		# get masks and targets
		y_inp, y_tar = y[:, :-1], y[:, 1:]
		enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)
		
		# devices
		x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask = to_devices(
			(x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask),
			device)

		# forward
		model.eval()
		with torch.no_grad():
			y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)
			loss = loss_fn(y_pred.permute(0, 2, 1), y_tar)

		# metrics
		batch_loss = loss.item()
		batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()

		return batch_loss, batch_acc

	# main training loop

	batch_losses, batch_accs = [], []
	epoch_losses, epoch_accs = [], []
	val_epoch_losses, val_epoch_accs = [], []
	for epoch in range(epochs):
		start_ = time.time()

		# train
		epoch_loss = 0.0
		epoch_acc = 0.0
		for i, (x, y) in enumerate(train_dataloader):

			batch_loss, batch_acc = train_step(x, y)
			
			batch_losses.append(batch_loss)
			batch_accs.append(batch_acc)

			epoch_loss += (batch_loss - epoch_loss) / (i + 1)
			epoch_acc += (batch_acc - epoch_acc) / (i + 1)
			
			if i % 50 == 0:
				print('Batch {} Loss {:.4f} Accuracy {:.4f} in {:.4f} s per batch'.format(
					i, epoch_loss, epoch_acc, (time.time() - start_)/(i+1)))
					
		epoch_losses.append(epoch_loss)
		epoch_accs.append(epoch_acc)
		
		# val
		if val_dataloader is not None:
			val_epoch_loss = 0.0
			val_epoch_acc = 0.0
			for i, (x, y) in enumerate(val_dataloader):
				batch_loss, batch_acc = val_step(x, y)
				val_epoch_loss += (batch_loss - val_epoch_loss) / (i + 1)
				val_epoch_acc += (batch_acc - val_epoch_acc) / (i + 1)
					
			val_epoch_losses.append(val_epoch_loss)
			val_epoch_accs.append(val_epoch_acc)
			
			print('Epoch {} Loss {:.4f} Accuracy {:.4f} Val Loss {:.4f} Val Accuracy {:.4f} in {:.4f} secs \n'.format(
				epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, time.time() - start_))
		else:
			print('Epoch {} Loss {:.4f} Accuracy {:.4f} in {:.4f} secs \n'.format(
				epoch, epoch_loss, epoch_acc, time.time() - start_))

	return epoch_losses, epoch_accs, val_epoch_losses, val_epoch_accs


if __name__ == "__main__":

	from utils.arguments import parser
	args = parser.parse_args()

	transformer_args = {
		'num_layers' : args.layers,
		'num_heads' : args.heads,
		'dff' : args.dff,
		'd_model' : args.d_model,
		'input_vocab_size' : args.vocab_size,
		'target_vocab_size' : args.vocab_size,
		'pe_input' : args.max_pe,
		'pe_target' : args.max_pe,
		'rate' : args.dropout}

	opt_args = {
		'lr' : args.lr}

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_dataloader, val_dataloader, test_dataloader = preprocess.load_and_preprocess(
		args.langs, args.batch_size, args.vocab_size, "ted_multi")

	train(device, args.epochs, transformer_args, opt_args, train_dataloader, val_dataloader=val_dataloader)
