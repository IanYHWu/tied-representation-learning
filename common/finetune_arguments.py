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
parser.add_argument('--checkpoint_location',
    default='..', type=str,
    help='Checkpoint location for first pivot model.'
)
parser.add_argument('--checkpoint_location_2',
    default='..', type=str,
    help='Checkpoint location for second pivot model.'
)
parser.add_argument('--split',
    default='test', type=str,
    help='Which dataset split to use: test for test, val for val,\
    combine for both.'
)
parser.add_argument('--seed',
    default=11, type=int,
    help='Random seed for run.'
)

#### Languages + preprocessing
parser.add_argument('--langs',
    default=['en', 'fr'], nargs='+',
    help='Ted language codes for languages to train on.'
)
parser.add_argument('--temp',
    default=0.7, type=float,
    help='Sampling temperature for relative language sizes.'
)
parser.add_argument('--max_len',
    default=100, type=int,
    help='Maximum length of sequences.'
)
parser.add_argument('--single_direction',
    action='store_true',
    help='Whether to only train on one way (for bilingual mode).'
)
parser.add_argument('--zero_shot',
    default=[], nargs='+',
    help='Directions to zero shot translate as a list of pairs \
    (both directions are excluded from training).'
)

#### Optimization
parser.add_argument('--batch_size',
    default=5, type=int,
    help='Dataloader batch size.'
) 
parser.add_argument('--train_steps',
    default=40000, type=int,
    help='Total number of training steps.'
)
parser.add_argument('--max_lr',
    default=3e-5, type=float,
    help='Finetuning maximum learning rate.'
)
parser.add_argument('--warmup_steps',
    default=2500, type=float,
    help='Warmup steps for learning rate.'
)
parser.add_argument('--dropout',
    default=0.3, type=float,
    help='Dropout and attention dropout rates.'
)
parser.add_argument('--label_smoothing',
    default=None, type=float,
    help='Amount of label smoothing to apply.'
)

#### Auxiliary
parser.add_argument('--auxiliary',
    action='store_true',
    help='Whether to use auxiliary loss.'
)
parser.add_argument('--frozen_layers',
    default=[], nargs='+', type=int,
    help='Indicies of frozen layers.'
)
parser.add_argument('--aux_strength',
    default=0.0, type=float,
    help='Relative strength of aux regularisation.'
)

#### Verbosity
parser.add_argument('--verbose',
    default=100, type=int,
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
parser.add_argument('--test_batches',
    default=None, type=int,
    help='Batches to test on (if None will test on all).'
)

#### Decoding
parser.add_argument('--num_beams',
    default=5, type=int,
    help='Number of beams for decoding.'
)
parser.add_argument('--length_penalty',
    default=1.0, type=float,
    help='Length penalty for decoding.'
)
