"""
Initialise a model according to the input params
"""

import models.base_transformer as base_transformer


def initialise_model(params, device):
    if params.model == 'base':
        model_kwargs = base_transformer.base_transformer_args(params)
        model = base_transformer.Transformer(**model_kwargs).to(device)
    else:
        raise Exception("Model class {} not yet implemented")

    return model


def initialise_optimiser(params):
    # custom initialiser for the optimiser - use for L.R. schedule
    pass



