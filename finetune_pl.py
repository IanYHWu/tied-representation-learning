""" Finetune MBart for MNMT on given langauges. """

import torch
import numpy as np
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from models.mbart import OurMBartModel
from common.data import MNMTDataModule
from common import data_logger as logging
from common.finetune_arguments import parser


def main(params):
    # setup
    pl.seed_everything(params.seed, workers=True)
    new_root_path, new_name = params.location, params.name
    logger = logging.TrainLogger(params)
    logger.make_dirs()
    logger.save_params()
    if params.wandb:
        pl_logger = WandbLogger(project='mnmt', entity='nlp-mnmt-project', group='finetuning')
    else:
        pl_logger = True

    # load data and model
    datamodule = MNMTDataModule(params.langs, params.batch_size, params.max_len, T=params.temp)
    model = OurMBartModel(max_lr=params.max_lr, warmup_steps=params.warmup_steps,
        dropout=params.dropout, aux_strength=params.aux_strength)

    # finetune
    trainer = pl.Trainer(gpus=-1, max_steps=params.train_steps,
        logger=pl_logger, deterministic=True, accelerator='ddp')
    trainer.fit(model, datamodule)

    # save model
    if params.save:
        logger.save_model(params.train_steps, model.mbart, optimizer, scheduler=scheduler)


if __name__ == '__main__':
    params = parser.parse_args()
    main(params)