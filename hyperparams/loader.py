"""
Load in hyperparameters
"""

import yaml


class Loader:
    def __init__(self, input_obj, check_custom=False):
        if isinstance(input_obj, dict):
            input_dict = input_obj
        else:
            input_dict = input_obj.__dict__
        custom_model = input_dict['custom_model']
        if custom_model and check_custom:
            with open('hyperparams/config.yml', 'r') as f:
                custom_params = yaml.safe_load(f)[custom_model]
            combined_dict = {**input_dict, **custom_params}
            self._generate_loader(combined_dict)
        else:
            self._generate_loader(input_dict)

    def _generate_loader(self, combined_dict):
        for key, val in combined_dict.items():
            setattr(self, key, val)


if __name__ == '__main__':
    input_obj = {'Herbster': 78, 'Maneesh': 85, 'Pontus': 90, 'Demian': 69, 'custom_model': 'basic'}
    params = Loader(input_obj, check_custom=True)
    print(params.Maneesh)
    print(params.epochs)

