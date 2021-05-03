import torch
import datasets

from common.preprocess.preprocess_utils import train_tokenizer, pad_sequence, detokenize, AddTargetTokens
from common.preprocess.ted_multi import load_ted_multi
from common.preprocess.opus100 import load_opus100

def load_and_preprocess(dataset_name, *args, **kwargs):
    dataset_fns = {
        'ted_multi' : load_ted_multi,
        'opus100' : load_opus100,
    }
    return dataset_fns[dataset_name](*args, **kwargs)