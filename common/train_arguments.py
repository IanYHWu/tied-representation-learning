"""
Arguments for Training Argparse
"""

import argparse

_train_parser = argparse.ArgumentParser(description='MNMT Input Params, Training')

##### required args
_train_parser.add_argument(
    '--name', required=True,
    type=str, help="Name of run"
)

##### general args
_train_parser.add_argument(
    '--location', default='..',
    type=str, help="Save folder location"
)
_train_parser.add_argument(
    '--checkpoint', action='store_true',
    help="Load from checkpoint"
)
_train_parser.add_argument(
    '--custom_model', default=None,
    type=str, help="Load a custom model"
)
_train_parser.add_argument(
    '--dataset', default="ted_multi",
    type=str, help='Dataset Name'
)
_train_parser.add_argument(
    '--model', default='base',
    type=str, help='Model to use'
)
_train_parser.add_argument(
    '--langs', nargs='+', default=['en', 'fr'],
    type=str, help='Languages to translate'
)
_train_parser.add_argument(
    '--verbose', default=50, type=int,
    help='Frequency to print batch results.'
)
_train_parser.add_argument(
    '--wandb', action='store_true',
    help='Record the run in weights and biases.'
)
_train_parser.add_argument(
    '--device', default='gpu', type=str,
    help='Device to use for training (gpu or cpu)'
)
_train_parser.add_argument(
    '--max_len', default=None,
    type=int, help='Clip sequences to maximum length.'
)
_train_parser.add_argument(
    '--excluded', nargs='+', default=[],
    type=str, help='Pairs of languages to exclude\
    from training. Should be passed as a list which\
    is converted to a list of tuples. E.g. to exclude\
    en-fr and en-de pass en fr en de.'
)
_train_parser.add_argument(
    '--tokenizer', nargs='+', default=None,
    type=str, help='Tokenizer(s) in pretrained to use.'
)


#### for no tf testing
_train_parser.add_argument(
    '--test_freq', default=None,
    type=int, help='Frequency of no tf testing.'
)
_train_parser.add_argument(
    '--test_batches', default=50,
    type=int, help='Number of batches to test without tf.'
)
_train_parser.add_argument(
    '--beam_length', default=4,
    type=int, help='Beam length for no tf testing.'
)
_train_parser.add_argument(
    '--alpha', default=0.0,
    type=float, help='Beam search hyperparameter.'
)
_train_parser.add_argument(
    '--beta', default=0.0,
    type=float, help='Beam search hyperparameter.'
)


##### distributed training args
_train_parser.add_argument(
    '--distributed', action='store_true',
    help='Distribute training over multiple gpus (if available)'
)
_train_parser.add_argument(
    '--nodes', default=1, type=int,
    help='Number of data loading workers.'
)
_train_parser.add_argument(
    '--nr', default=0, type=int,
    help='Ranking within the nodes.'
)
_train_parser.add_argument(
    '--gpus', default=1, type=int,
    help='Number of gpus per node.'
)

##### pivot args
_train_parser.add_argument(
    '--pivot', action='store_true',
    help='Train a bilingual model that is part of a pivot'
)
_train_parser.add_argument(
    '--pivot_inds', nargs='+', default=[0, 1],
    type=int,
    help='Pivot indices - e.g. if langs=[de, en, fr], then training the de -> en pivot will require pivot_inds=(0, 1)'
)
_train_parser.add_argument(
    '--pivot_tokenizer_path', default=None, type=str,
    help='Path to multilingual tokeniser for pivot training'
)

##### auxiliary loss args
_train_parser.add_argument(
    '--auxiliary', action='store_true',
    help="Use auxiliary loss on encoder output."
)
_train_parser.add_argument(
    '--frozen_layers', nargs='+', default=[],
    type=int, help='Encoder layers frozen for grad wrt aux loss.'
)
_train_parser.add_argument(
    '--aux_strength', default=1.0, type=float,
    help='Strength of auxiliary loss relative to main loss.'
)

##### hyperparameters
_train_parser.add_argument(
    '--vocab_size', default=2000, type=int,
    help='Vocab size'
)
_train_parser.add_argument(
    '--batch_size', default=32, type=int,
    help='Batch Size'
)
_train_parser.add_argument(
    '--layers', default=2, type=int,
    help='Number of transformer layers'
)
_train_parser.add_argument(
    '--heads', default=2, type=int,
    help='Number of multi-attention heads'
)
_train_parser.add_argument(
    '--dff', default=128, type=int,
    help='dff'
)
_train_parser.add_argument(
    '--d_model', default=32, type=int,
    help='Decoder dimension'
)
_train_parser.add_argument(
    '--max_pe', default=1000, type=int,
    help='Max pe'
)
_train_parser.add_argument(
    '--dropout', default=0.1, type=float,
    help='Dropout rate'
)
_train_parser.add_argument(
    '--epochs', default=10, type=int,
    help='Epochs to train'
)
_train_parser.add_argument(
    '--warmup_steps', default=1000, type=float,
    help='Warmup step for lr'
)
_train_parser.add_argument(
    '--lr_scale', default=1.0, type=float,
    help='Scale for learning rate schedule'
)
_train_parser.add_argument(
    '--add_epochs', default=0, type=int,
    help='Add epochs to train. Used for checkpointing'
)

# for extra printing
_train_parser.add_argument(
    '--FLAGS', action='store_true',
    help='print where in the training cycle the script is.'
)

class train_parser:

    @staticmethod
    def parse_args():
        args = _train_parser.parse_args()

        # add on parse methods here
        args.excluded = [(args.excluded[i], args.excluded[i+1]) for i in range(0, len(args.excluded), 2)]
        if args.max_len is None:
            args.max_len = args.max_pe

        return args
