# NLP_project
NLP Project, UCL MSc 2021

To train a model, run
```
python train.py --name='my_model' 
```
from the command line. See ```common.train_arguments.py``` for a list of optional arguments.

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

To use distributed training add the argument ```--distributed``` and then set ```--nodes``` and ```---gpus``` to the number of nodes and gpus you have available and ``` -nr ``` to the node number (for ranking). Ensure that the environment variables ```MASTER_ADDR``` and ```MASTER_PORT``` have been set before running train.py.

To test a model, run
```
python test.py --name='my_model' --test_name='my_test' --custom_model='model_name'
```
This will automatically create a ```test``` folder inside the training folder, where the test logs will be stored. See
```common.test_arguments.py``` for more testing options. 

