"""
Arguments for Testing Argparse
"""

import argparse

_test_parser = argparse.ArgumentParser(description='MNMT Input Params, Testing')

_test_parser.add_argument(
    '--name', required=True,
    type=str, help="Name of model"
)
_test_parser.add_argument(
    '--test_name', required=True,
    type=str, help="Name of test instance"
)
_test_parser.add_argument(
    '--location', default='..',
    type=str, help="Save folder location"
)
_test_parser.add_argument(
    '--custom_model', default=None,
    type=str, help="Load a custom model"
)
_test_parser.add_argument(
    '--dataset', default="ted_multi",
    type=str, help='Dataset Name'
)
_test_parser.add_argument(
    '--langs', nargs='+', default=['en', 'fr'],
    type=str, help='Languages to translate'
)
train_parser.add_argument(
    '--excluded', nargs='+', default=[],
    type=str, help='Pairs of languages to exclude\
    from training. Should be passed as a list which\
    is converted to a list of tuples. E.g. to exclude\
    en-fr and en-de pass en fr en de.')
_test_parser.add_argument(
    '--teacher_forcing', action='store_true',
    help='Teacher forcing for inference'
)
_test_parser.add_argument(
    '--batch_size', default=20, type=int,
    help='Batch size for datasets'
)
_test_parser.add_argument(
    '--vocab_size', default=2000, type=int,
    help='Vocab size for tokenizers'
)
_test_parser.add_argument(
    '--beam_length', default=1, type=int,
    help='1 for greedy inference. >1 for n-beam search.'
)
_test_parser.add_argument(
    '--verbose', default=50, type=int,
    help='Frequency to print batch results.'
)
_test_parser.add_argument(
    '--pivot', action='store_true',
    help='Test a multilingual pivot model'
)
_test_parser.add_argument(
    '--pivot_tokenizer_path', default=None, type=str,
    help='Path to multilingual tokeniser for pivot testing'
)
_test_parser.add_argument(
    '--pivot_model_1', default=None, type=str,
    help='Path to pivot model 1'
)
_test_parser.add_argument(
    '--pivot_model_2', default=None, type=str,
    help='Path to pivot model 2'
)

class test_parser:

    @staticmethod
    def parse_args():
        args = _test_parser.parse_args()

        # add on parse methods here
        args.excluded = [(args.excluded[i], args.excluded[i+1]) for i in range(0, len(args.excluded), 2)]

        return args



