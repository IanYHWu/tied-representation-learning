""" LR schedulers """

import torch 

class WarmupDecay(torch.optim.lr_scheduler._LRScheduler):
	""" Learning rate decay with warmup steps as in Attention is All You Need."""

	def __init__(self, optimizer, warmup_steps, d_model, lr_scale = 1.0):
		self.warmup_steps = warmup_steps
		self.d_model = d_model
		self.lr_scale = lr_scale
		self.n_lrs = len(optimizer.param_groups)
		super().__init__(optimizer)

	def get_lr(self):
		t = self._step_count
		lr = (self.d_model ** -0.5) * min(t ** -0.5, t * (self.warmup_steps ** -1.5))
		return [self.lr_scale * lr for _ in range(self.n_lrs)]

if __name__ == '__main__':

	model = torch.nn.Linear(5, 2)

	optimizer = torch.optim.Adam(model.parameters())
	scheduler = WarmupDecay(optimizer, 100, 20)

	lrs = [optimizer.param_groups[0]['lr']]

	for _ in range(1000):
		scheduler.step()
		lrs.append(optimizer.param_groups[0]['lr'])

	import matplotlib.pyplot as plt 
	plt.plot(lrs)
	plt.show()
