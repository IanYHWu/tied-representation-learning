"""
Arguments for Training Argparse
"""

import argparse

train_parser = argparse.ArgumentParser(description='MNMT Input Params, Training')

train_parser.add_argument(
    '--name', required=True,
    type=str, help="Name of run"
)
train_parser.add_argument(
    '--location', default='..',
    type=str, help="Save folder location"
)
train_parser.add_argument(
    '--checkpoint', action='store_true',
    help="Load from checkpoint"
)
train_parser.add_argument(
    '--custom_model', default=None,
    type=str, help="Load a custom model"
)
train_parser.add_argument(
    '--dataset', default="ted_multi",
    type=str, help='Dataset Name'
)
train_parser.add_argument(
    '--model', default='base',
    type=str, help='Model to use'
)
train_parser.add_argument(
    '--langs', nargs='+', default=['en', 'fr'],
    type=str, help='Languages to translate'
)
train_parser.add_argument(
    '--vocab_size', default=2000, type=int,
    help='Vocab size'
)
train_parser.add_argument(
    '--batch_size', default=32, type=int,
    help='Batch Size'
)
train_parser.add_argument(
    '--layers', default=2, type=int,
    help='Number of transformer layers'
)
train_parser.add_argument(
    '--heads', default=2, type=int,
    help='Number of multi-attention heads'
)
train_parser.add_argument(
    '--dff', default=128, type=int,
    help='dff'
)
train_parser.add_argument(
    '--d_model', default=32, type=int,
    help='Decoder dimension'
)
train_parser.add_argument(
    '--max_pe', default=1000, type=int,
    help='Max pe'
)
train_parser.add_argument(
    '--dropout', default=0.1, type=float,
    help='Dropout rate'
)
train_parser.add_argument(
    '--epochs', default=10, type=int,
    help='Epochs to train'
)
train_parser.add_argument(
    '--warmup_steps', default=1000, type=float,
    help='Warmup step for lr'
)
train_parser.add_argument(
    '--lr_scale', default=1.0, type=float,
    help='Scale for learning rate schedule'
)
train_parser.add_argument(
    '--add_epochs', default=0, type=int,
    help='Add epochs to train. Used for checkpointing'
)
train_parser.add_argument(
    '--auxiliary', action='store_true',
    help="Use auxiliary loss on encoder output."
)
train_parser.add_argument(
    '--frozen_layers', nargs='+', default=[],
    type=int, help='Encoder layers frozen for grad wrt aux loss.'
)
train_parser.add_argument(
    '--aux_strength', default=1.0, type=float,
    help='Strength of auxiliary loss relative to main loss.'
)
train_parser.add_argument(
    '--verbose', default=50, type=int,
    help='Frequency to print batch results.'
)

