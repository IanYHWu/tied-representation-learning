""" Argument parser for finetune.py """

import argparse
parser = argparse.ArgumentParser()

#### General
parser.add_argument('--name',
    required=True, type=str,
    help='Name of run.'
)
parser.add_argument('--location',
    default='..', type=str,
    help='File path for saving results.'
)

#### Languages + preprocessing
parser.add_argument('--langs',
    default=['en', 'fr'], nargs='+',
    help='Ted language codes for languages to train on.'
)
parser.add_argument('--temp',
    default=1.0, type=float,
    help='Sampling temperature for relative language sizes.'
)
parser.add_argument('--max_len',
    default=100, type=int,
    help='Maximum length of sequences.'
)
parser.add_argument('--single_directions',
    action='store_true',
    help='Wether to only train on one way (for bilingual mode).'
)

#### Optimization
parser.add_argument('--batch_size',
    default=6, type=int,
    help='Dataloader batch size.'
) 
parser.add_argument('--train_steps',
    default=5000, type=int,
    help='Total number of training steps.'
)
parser.add_argument('--max_lr',
    default=3e-5, type=float,
    help='Finetuning maximum learning rate.'
)
parser.add_argument('--warmup_steps',
    default=2500, type=float,
    help='Finetuning maximum learning rate.'
)
parser.add_argument('--dropout',
    default=0.3, type=float,
    help='Dropout and attention dropout rates.'
)

#### Verbosity
parser.add_argument('--verbose',
    default=20, type=int,
    help='Frequency to print during training.'
)
parser.add_argument('--wandb',
    action='store_true',
    help='wether to log results to wandb.'
)
parser.add_argument('--save',
    action='store_true',
    help='wether to save model after training.'
)

#### Decoding
parser.add_argument('--num_beams',
    default=5, type=int,
    help='Number of beams for decoding.'
)
parser.add_argument('--length_penalty',
    default=0.6, type=float,
    help='Length penalty for decoding.'
)