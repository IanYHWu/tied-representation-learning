"""Test lr scheduler modeule."""

import torch 
from hyperparams.schedule import WarmupDecay

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

print(scheduler._step_count)
scheduler.save('./scheduler.json')
new_opt = torch.optim.Adam(model.parameters())
new_scheduler = WarmupDecay(new_opt, 100, 20)
new_scheduler.load('./scheduler.json')
print(new_scheduler._step_count)