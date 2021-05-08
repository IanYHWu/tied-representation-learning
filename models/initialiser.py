"""
Initialise a model according to the input params
"""

import models.base_transformer as base_transformer


def initialise_model(params, device):
    if params.model == 'base':
        model_kwargs = base_transformer.base_transformer_args(params)
        if len(params.langs) > 2:
        	model_kwargs['vocab_size'] = model_kwargs['input_vocab_size']
        	model_kwargs.pop('input_vocab_size', None)
        	model_kwargs.pop('target_vocab_size', None)
        	model = base_transformer.MultiTransformer(**model_kwargs).to(device)
        else:
        	model = base_transformer.Transformer(**model_kwargs).to(device)
    else:
        raise Exception("Model class {} not yet implemented")

    return model




