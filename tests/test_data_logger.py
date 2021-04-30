""" Test data logger module. """

import common.data_logger as logging


# create a basic model, optimiser and scheduler and test loading and saving

import torch
from models.initialiser import initialise_model
from hyperparams.schedule import WarmupDecay

# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class params:
	model = 'base'
	d_model = 10
	dff = 20
	layers = 2
	heads = 2
	max_pe = 1000
	vocab_size = 100
	dropout = 0.1
	location = '.'
	name = 'test_logging'
model = initialise_model(params, device)
optimizer = torch.optim.Adam(model.parameters())
scheduler = WarmupDecay(optimizer, params.d_model, 1000)
epoch = 10

# test logger
logger = logging.TrainLogger(params)
logger.make_dirs()
logger.save_model(epoch, model, optimizer, scheduler=scheduler)


model2 = initialise_model(params, device)
optimizer2 = torch.optim.Adam(model2.parameters())
scheduler2 = WarmupDecay(optimizer2, params.d_model, 1000)

path = './test_logging/checkpoint/checkpoint'
model2, optimizer2, epoch, scheduler = logging.load_checkpoint(path, device, model,
	optimizer=optimizer, scheduler=scheduler)

print(epoch)





