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
    '--tokenizer', nargs='+', default=None,
    type=str, help='Tokenizer(s) in pretrained to use.'
)
_test_parser.add_argument(
    '--pivot', action='store_true',
    help='Test a multilingual pivot model'
)
_test_parser.add_argument(
    '--pivot_tokenizer_path_1_1', default=None, type=str,
    help='Path to bilingual tokeniser 1, language pair 1 for pivot testing'
)
_test_parser.add_argument(
    '--pivot_tokenizer_path_1_2', default=None, type=str,
    help='Path to bilingual tokeniser 2, language pair 1 for pivot testing'
)
_test_parser.add_argument(
    '--pivot_tokenizer_path_2_1', default=None, type=str,
    help='Path to bilingual tokeniser 1, language pair 2 for pivot testing'
)
_test_parser.add_argument(
    '--pivot_tokenizer_path_2_2', default=None, type=str,
    help='Path to bilingual tokeniser 2, language pair 2 for pivot testing'
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
        args = __test_parser.parse_args()

        # add on parse methods here
        args.excluded = [(args.excluded[i], args.excluded[i+1]) for i in range(0, len(args.excluded), 2)]

        return args



