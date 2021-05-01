"""
Pipeline for loading translation dataset via HuggingFace, training BPE
tokenizers for given languages and creating a torch dataloader.

If doing multi-lingual translation then a single, shared tokenizer is trained 
however for bilingual translation individual tokenizers are used.

For a dataset to be loadable it needs a filter_languages function which will
extract the required translations from each example.
"""

import torch
import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from common.preprocess.preprocess_utils import train_tokenizer, pad_sequence, detokenize





if __name__ == "__main__":
    # !!!! very large download !!!
    # train, val, test, tokenizers = load_and_preprocess(['en', 'fr', 'de'], 32, 10000, "ted_multi", multi=True)
    train, val, test, tokenizers = load_and_preprocess(['en', 'fr'], 32, 10000, "ted_multi", multi=False)

    for i, data in enumerate(val):
        x, y = data
        print(detokenize(x, tokenizers[0]))
        break

