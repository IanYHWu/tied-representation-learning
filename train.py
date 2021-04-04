"""
Training Loop for MNMT
"""

import torch
import models.base_transformer as base_transformer
import utils.preprocess as preprocess
from utils.arguments import parser
import time
import utils.train_logger as logging
from hyperparams.loader import Loader
import models.initialiser as initialiser


def to_devices(tensors, device):
	return (tensor.to(device) for tensor in tensors)


def loss_fn(y_pred, y_true, criterion):
	_mask = torch.logical_not(y_true == 0).float()
	_loss = criterion(y_pred, y_true)
	return (_loss * _mask).sum() / _mask.sum()


def accuracy_fn(y_pred, y_true):
	_mask = torch.logical_not(y_true == 0).float()
	_acc = (torch.argmax(y_pred, axis=-1) == y_true)
	return (_acc * _mask).sum() / _mask.sum()


def train_step(x, y, model, criterion, optimizer, device):
	# get masks and targets
	y_inp, y_tar = y[:, :-1], y[:, 1:]
	enc_mask, look_ahead_mask, dec_mask = base_transformer.create_masks(x, y_inp)

	# devices
	x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask = to_devices(
		(x, y_inp, y_tar, enc_mask, look_ahead_mask, dec_mask),
		device)

	# forward
	model.train()
	y_pred, _ = model(x, y_inp, enc_mask, look_ahead_mask, dec_mask)
	loss = loss_fn(y_pred.permute(0, 2, 1), y_tar, criterion)

	# backward
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# metrics
	batch_loss = loss.cpu().item()
	batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()

	return batch_loss, batch_acc


def val_step(x, y, model, criterion, device):
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
		loss = loss_fn(y_pred.permute(0, 2, 1), y_tar, criterion)

	# metrics
	batch_loss = loss.item()
	batch_acc = accuracy_fn(y_pred.detach(), y_tar).cpu().item()

	return batch_loss, batch_acc


def train(device, params, train_dataloader, val_dataloader=None):
	"""Training Loop"""

	new_root_path = params.location
	new_name = params.name
	if params.checkpoint:
		params = logging.load_params(new_root_path + '/' + new_name)
		params.location = new_root_path
		params.name = new_name
		logger = logging.Logger(params)
		logger.make_dirs()
	else:
		logger = logging.Logger(params)
		logger.make_dirs()
	logger.save_params()

	model = initialiser.initialise_model(params, device)
	optimizer = torch.optim.Adam(model.parameters(), params.lr)
	criterion = torch.nn.CrossEntropyLoss(reduction='none')
	epoch = 0
	if params.checkpoint:
		model, optimizer, epoch = logging.load_checkpoint(logger.checkpoint_path, model, optimizer)

	batch_losses, batch_accs = [], []
	epoch_losses, epoch_accs = [], []
	val_epoch_losses, val_epoch_accs = [], []

	while epoch < params.epochs:
		start_ = time.time()

		# train
		epoch_loss = 0.0
		epoch_acc = 0.0
		val_epoch_loss = 0.0
		val_epoch_acc = 0.0
		for i, (x, y) in enumerate(train_dataloader):

			batch_loss, batch_acc = train_step(x, y, model, criterion, optimizer, device)
			
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
			for i, (x, y) in enumerate(val_dataloader):
				batch_loss, batch_acc = val_step(x, y, model, criterion, device)
				val_epoch_loss += (batch_loss - val_epoch_loss) / (i + 1)
				val_epoch_acc += (batch_acc - val_epoch_acc) / (i + 1)

			val_epoch_losses.append(val_epoch_loss)
			val_epoch_accs.append(val_epoch_acc)

			print('Epoch {} Loss {:.4f} Accuracy {:.4f} Val Loss {:.4f} Val Accuracy {:.4f} in {:.4f} secs \n'.format(
				epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, time.time() - start_))
		else:
			print('Epoch {} Loss {:.4f} Accuracy {:.4f} in {:.4f} secs \n'.format(
				epoch, epoch_loss, epoch_acc, time.time() - start_))

		epoch += 1

		logger.save_model(epoch, model, optimizer)
		logger.log_results([epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc])

	return epoch_losses, epoch_accs, val_epoch_losses, val_epoch_accs


if __name__ == "__main__":

	args = parser.parse_args()

	# Loader can also take in any dictionary of parameters
	params = Loader(args, check_custom=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_dataloader, val_dataloader, test_dataloader = preprocess.load_and_preprocess(
		params.langs, params.batch_size, params.vocab_size, params.dataset)

	train(device, params, train_dataloader, val_dataloader=val_dataloader)
