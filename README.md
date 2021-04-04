# NLP_project
NLP Project, UCL MSc 2021

To train a model, run
```
python train.py --name='my_model' 
```
from the command line. See ```utils.arguments.py``` for a list of optional arguments.

A folder containing the outputs of the training run will be automatically created, with the default location of this 
folder being ```..``` relative to the project root. The name of this folder is given by the required ```--name``` 
command line argument.

To continue training from a checkpoint, run
```
python train.py --name='my_model' --checkpoint=True
```
from the command line. 

Custom parameters can be defined in ```hyperparams/config.yml```. These can be trained using
```
python train.py --name='my_model' --custom_model='model_name'
```
Parameters defined in ```hyperparams/config.yml``` overwrite any parameters defined from the command line.

## TODO
* Inference loop
* Test: train.py, test.py, etc.
* Hyperparameter folder
