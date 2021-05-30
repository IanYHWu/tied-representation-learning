# Improving Zero-Shot Performance in Pretrained Translation Models through Tied Representation Learning
### Hamish Scott, Ian Wu, Daniel Shani, Dhruv Kaul
Abstract
> Unsupervised pretraining has been shown to improve the performance of multilingual Neural Machine Translation (MNMT) in the supervised setting, but how it impacts performance in the traditionally-challenging zero-shot setting remains an open question. In this paper, we investigate the performance of the well-established pretrained model mBART50 in the zero-shot setting, and improve finetuning by introducing language invariance through *tied representation learning*, based on the work of Arivazhagan et al. (2019a). Our experiments in the low-resource regime show that vanilla mBART50 performs worse than pivoting in the zero-shot setting. Tied representations are able to improve performance by nearly 10 BLEU, surpassing the performance of pivoting while having a negligible impact on the supervised performance. By altering the tying strength, we are able improve this by a further 5 BLEU, surpassing the performance of supervised finetuning. Finally, we extend tied representation learning by applying it to selected encoder layers only, but find that this reduces both zero-shot and supervised performance.

To finetune mBart50 for multilingual translation (for example for English-Turkish-Azerbijani translation):
```
python train.py --name='my_model' --langs en tr az
```
See ```common/finetune_arguments.py``` for a list of optional arguments.

To test your model:
```
python test.py --name='my_model' --langs en tr az
```

### References
Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Roee Aharoni, Melvin Johnson, and Wolfgang Macherey.2019a.   The missing ingredient in zero-shot neural machine translation.
