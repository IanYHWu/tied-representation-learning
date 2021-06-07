""" Finetune MBart for MNMT on given langauges. """

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from transformers import MBartForConditionalGeneration
from common.data import MNMTDataModule
from common import data_logger as logging
from hyperparams.schedule import WarmupDecay


class OurMBartModel(pl.LightningModule):

    def __init__(self, pretrained_name='facebook/mbart-large-50', max_lr=3e-5, warmup_steps=2500,
        dropout=0.3, aux_strength=0.0):
        super().__init__()
        self.mbart = MBartForConditionalGeneration.from_pretrained(pretrained_name)
        self.max_lr = max_lr 
        self.warmup_steps = warmup_steps
        self.aux_strength = aux_strength

        self.mbart.config.dropout = dropout 
        self.mbart.config.attention_dropout = dropout

    def forward(self, *args, **kwargs):
        return self.mbart(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.log('ce_loss', output.loss)
        self.log('acc', output.acc)
        self.log('aux', output.aux_loss)
        return output.loss + self.aux_strength * output.aux_loss

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.log('val_ce_loss', output.loss)
        self.log('val_acc', output.acc)
        self.log('val_aux', output.aux)

    def shared_step(self, batch, batch_idx):
        input_ids = batch['input_ids_en']
        y = batch['input_ids_cs']
        decoder_input_ids = y[:,:-1].contiguous()
        y_tar = y[:,1:].contiguous()
        attention_mask = input_ids!=0
        decoder_attention_mask = decoder_input_ids!=0

        # cross-entropy
        output = self.mbart(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            labels=y_tar)
        output.acc = ((output.logits.argmax(-1)==y_tar)*decoder_attention_mask).sum() / decoder_attention_mask.sum()

        # tie representations
        x_enc = output.encoder_last_hidden_state
        y_enc = self.mbart.model.encoder(decoder_input_ids, attention_mask=dec_mask)['last_hidden_state']
        x_enc = torch.max(x_enc + -999 * (1-attention_mask.type(x_enc.dtype)).unsqueeze(-1), dim=1)[0]
        y_enc = torch.max(y_enc + -999 * (1-decoder_attention_mask.type(y_enc.dtype)).unsqueeze(-1), dim=1)[0]
        output.aux_loss = 1.0 - F.cosine_similarity(x_enc, y_enc)

        return output

    def configure_optimizers(self):
        lr_scale = self.max_lr * np.sqrt(self.warmup_steps)
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = WarmupDecay(optimizer, self.warmup_steps, 1, lr_scale=lr_scale)
        return [optimizer], [scheduler]


def main(params):
    # setup logging
    new_root_path, new_name = params.location, params.name
    logger = logging.TrainLogger(params)
    logger.make_dirs()
    logger.save_params()

    # seed and wandb
    pl.seed_everything(params.seed, workers=True)
    wandb_logger = WandbLogger(project='mnmt', entity='nlp-mnmt-project', group='finetuning')

    # load data and model
    tokenizer, sampler, test_loaders = load_all(params)

    datamodule = MNMTDataModule(params.langs, params.batch_size, params.max_len, T=params.temp)
    model = OurMBartModel(max_lr=params.max_lr, warmup_steps=params.warmup_steps, dropout=params.dropout)

    # finetune
    trainer = pl.Trainer(gpus=params.gpus, max_steps=params.train_steps, logger=wandb_logger, deterministic=True)
    trainer.fit(model, datamodule)

    # save model
    if params.save:
        logger.save_model(params.train_steps, model.mbart, optimizer, scheduler=scheduler)

