"""
Arguments for Argparse
"""

import argparse

parser = argparse.ArgumentParser(description='MNMT Input Params')

parser.add_argument(
    '--name', required=True,
    type=str, help="Name of run"
)
parser.add_argument(
    '--location', default='..',
    type=str, help="Save folder location"
)
parser.add_argument(
    '--checkpoint', default=False,
    type=bool, help="Load from checkpoint"
)
parser.add_argument(
    '--custom_model', default=None,
    type=str, help="Load a custom model"
)
parser.add_argument(
    '--dataset', default="ted_multi",
    type=str, help='Dataset Name'
)
parser.add_argument(
    '--model', default='base',
    type=str, help='Model to use'
)
parser.add_argument(
    '--langs', nargs='+', default=['en', 'fr'],
    type=str, help='Languages to translate'
)
parser.add_argument(
    '--vocab_size', default=2000, type=int,
    help='Vocab size'
)
parser.add_argument(
    '--batch_size', default=32, type=int,
    help='Batch Size'
)
parser.add_argument(
    '--layers', default=2, type=int,
    help='Number of transformer layers'
)
parser.add_argument(
    '--heads', default=2, type=int,
    help='Number of multi-attention heads'
)
parser.add_argument(
    '--dff', default=128, type=int,
    help='dff'
)
parser.add_argument(
    '--d_model', default=32, type=int,
    help='Decoder dimension'
)
parser.add_argument(
    '--max_pe', default=1000, type=int,
    help='Max pe'
)
parser.add_argument(
    '--dropout', default=0.1, type=float,
    help='Dropout rate'
)
parser.add_argument(
    '--epochs', default=10, type=int,
    help='Epochs to train'
)
parser.add_argument(
    '--warmup_steps', default=1000, type=float,
    help='Warmup step for lr'
)
parser.add_argument(
    '--lr_scale', default = 1.0, type=float,
    help='Scale for learning rate schedule'
)

