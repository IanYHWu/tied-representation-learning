import torch
import datasets

from common.preprocess.preprocess_utils import train_tokenizer, pad_sequence, detoknize, AddTargetTokens
from common.preprocess.ted_multi import load_ted_multi

def load_and_preprocess(dataset_name, *args, **kwargs):
    dataset_fns = {
        'ted_multi' : load_ted_multi
    }
    return dataset_fns[dataset_name](*args, **kwargs)