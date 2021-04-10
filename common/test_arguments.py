"""
Arguments for Testing Argparse
"""

import argparse

test_parser = argparse.ArgumentParser(description='MNMT Input Params, Testing')

test_parser.add_argument(
    '--name', required=True,
    type=str, help="Name of model"
)
test_parser.add_argument(
    '--test_name', required=True,
    type=str, help="Name of test instance"
)
test_parser.add_argument(
    '--location', default='..',
    type=str, help="Save folder location"
)
test_parser.add_argument(
    '--custom_model', default=None,
    type=str, help="Load a custom model"
)
test_parser.add_argument(
    '--dataset', default="ted_multi",
    type=str, help='Dataset Name'
)
test_parser.add_argument(
    '--langs', nargs='+', default=['en', 'fr'],
    type=str, help='Languages to translate'
)
test_parser.add_argument(
    '--teacher_forcing', default=False, type=bool,
    help='Teacher forcing for inference'
)

