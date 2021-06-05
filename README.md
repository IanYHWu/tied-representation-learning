# Improving Zero-Shot Performance in Pretrained Translation Models through Tied Representation Learning
### Hamish Scott, Ian Wu, Daniel Shani, Dhruv Kaul

This repository accompanies the project "Improving Zero-Shot Performance in Pretrained Translation Models through Tied 
Representation Learning". In this project, we investigate the performance of mBART50 in the zero-shot setting, and propose
a training method to improve it. A copy of the report can be found in `tied_representation_learning.pdf`.

Dataset: TED-multi https://huggingface.co/datasets/ted_multi

To finetune mBART50 for multilingual translation (Az/Tr/En) without TRL:
```
python finetune.py --name='my_model' --langs en az tr --train_steps 40000 --save --auxiliary --aux_strength 1.0 --zero_shot az tr
```
To finetune with TRL:
```
python finetune.py --name='my_model' --langs en az tr --train_steps 40000 --save --auxiliary --aux_strength 1.0 --zero_shot az tr
```
The `--auxiliary` flag indicates to use TRL, with tying strength given by `--aux_strength`. The `--zero_shot` flag indicates
which directions to treat as the zero-shot directions.

To finetune with TRL applied only to specific encoder layers:
```
python finetune.py --name='my_model' --langs en az tr --train_steps 40000 --save --auxiliary --aux_strength 1.0 --zero_shot az tr --frozen_layers 0, 1, 2
```
The `--frozen_layers` flag indicates which encoder layers to train without using the auxiliary loss 

See ```common/finetune_arguments.py``` for a list of optional arguments.

To test a finetuned model:
```
python finetune_test.py --name='my_model' --langs en az tr --location 'my_location' 
```
Testing uses beam search by default, with a beam width of 5.

To use TRL on a model trained from scratch:
```
python train.py --name='my_model' --langs en az tr --location 'my_location' --custom_model='my_model --auxiliary'
```
where the `--custom_model` flag indicates which set of hyperparameters, contained in `hyperparams/config.yml`, to use

To test a model trained from scratch:
```
python test.py --name='my_model' --langs en az tr --location 'my_location' --custom_model='my_model'
```


### References
Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Roee Aharoni, Melvin Johnson, and Wolfgang Macherey.2019a.   The missing ingredient in zero-shot neural machine translation.

Ye Qi, Devendra Singh Sachan, Matthieu Felix, Sar- guna Janani Padmanabhan, and Graham Neubig. 2018. When and why are pre-trained word embed- dings useful for neural machine translation?

Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Na- man Goyal, Vishrav Chaudhary, Jiatao Gu, and An- gela Fan. 2020. Multilingual translation with exten- sible multilingual pretraining and finetuning.