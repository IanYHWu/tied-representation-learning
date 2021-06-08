""" Finetune MBart for MNMT on given langauges. """
import torch
import torch.nn.functional as F 
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import wandb
from transformers import MBartForConditionalGeneration
import time

from common.utils import accuracy_fn, to_devices
from common.data import MNMTDataModule
from common import data_logger as logging
from hyperparams.schedule import WarmupDecay


def main(params):
    """ Finetunes the mBart50 model on a selection of languages. """
    pl.seed_everything(params.seed, workers=True)

    if params.wandb:
        wandb.init(project='mnmt', entity='nlp-mnmt-project', group='finetuning',
            config={k: v for k, v in params.__dict__.items() if isinstance(v, (float, int, str, list))})

    new_root_path = params.location
    new_name = params.name
    logger = logging.TrainLogger(params)
    logger.make_dirs()
    logger.save_params()

    # get data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datamodule = MNMTDataModule(params.langs, params.batch_size, params.max_len, T=params.temp)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')

    # model and optimizer
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)
    optimizer = torch.optim.Adam(model.parameters())
    lr_scale = params.max_lr * np.sqrt(params.warmup_steps)
    scheduler = WarmupDecay(optimizer, params.warmup_steps, 1, lr_scale=lr_scale)
    model.config.dropout = params.dropout 
    model.config.attention_dropout = params.dropout

    def freeze_layers(layers, unfreeze=False):
        for n in layers:
            for parameter in model.model.encoder.layers[n].parameters():
                parameter.requires_grad = unfreeze

    def train_step(x, y, aux=False):

        y_inp, y_tar = y[:,:-1].contiguous(), y[:,1:].contiguous()
        enc_mask, dec_mask = (x != 0), (y_inp != 0)

        x, y_inp, y_tar, enc_mask, dec_mask = to_devices(
          (x, y_inp, y_tar, enc_mask, dec_mask), device)

        model.train()
        if aux: freeze_layers(params.frozen_layers, unfreeze=True)
        output = model(input_ids=x, decoder_input_ids=y_inp,
                   labels=y_tar, attention_mask=enc_mask,
                   decoder_attention_mask=dec_mask)
        optimizer.zero_grad()
        loss = output.loss
        loss.backward(retain_graph=aux)

        if aux: freeze_layers(params.frozen_layers)
        torch.set_grad_enabled(aux)

        x_enc = output.encoder_last_hidden_state
        y_enc = model.model.encoder(y_inp, attention_mask=dec_mask)['last_hidden_state']
        x_enc = torch.max(x_enc + -999 * (1-enc_mask.type(x_enc.dtype)).unsqueeze(-1), dim=1)[0]
        y_enc = torch.max(y_enc + -999 * (1-dec_mask.type(y_enc.dtype)).unsqueeze(-1), dim=1)[0]
        aux_loss = (1.0 - F.cosine_similarity(x_enc, y_enc)).mean()
        scaled_aux_loss = params.aux_strength * aux_loss
        
        torch.set_grad_enabled(True)
        if aux: scaled_aux_loss.backward()

        optimizer.step()
        scheduler.step()

        accuracy = accuracy_fn(output.logits, y_tar)

        return loss.item(), aux_loss.item(), accuracy.item()

    losses, aux_losses, accs = [], [], []
    start_ = time.time()
    for i, (x, y) in enumerate(datamodule.train_dataloader()):
        if i >= params.train_steps:
            break
       
        loss, aux_loss, acc = train_step(x, y, aux=params.auxiliary)
        losses.append(loss)
        aux_losses.append(aux_loss)
        accs.append(acc)

        if i % params.verbose == 0:
            print('Batch {} Loss {:.4f} Aux Loss {:.4f} Acc {:.4f} in {:.4f} secs per batch'.format(
                i, np.mean(losses[-params.verbose:]), np.mean(aux_losses[-params.verbose:]),
                np.mean(accs[-params.verbose:]), (time.time() - start_)/(i+1)))
        if params.wandb:
            wandb.log({'train_loss':loss, 'aux_loss':aux_loss, 'train_acc':acc})

    # save results
    if params.save:
        logger.save_model(params.train_steps, model, optimizer, scheduler=scheduler)
    
    train_results = {'loss':[np.mean(losses)], 'aux_loss':[np.mean(aux_losses)], 'accuarcy':[np.mean(accs)]}
    pd.DataFrame(train_results).to_csv(logger.root_path + '/train_results.csv', index=False)

    if params.wandb:
        wandb.finish()


if __name__ == '__main__':

    from common.finetune_arguments import parser
    params = parser.parse_args()
    main(params)

