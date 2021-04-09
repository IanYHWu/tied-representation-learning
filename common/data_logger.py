"""
Writing useful things to disk
"""

import os
import pandas as pd
import torch
import numpy as np
import json
from hyperparams.loader import Loader
from common.preprocess import detokenize


class TrainLogger:
    def __init__(self, params):
        self.params = params
        self.root = self.params.location
        self.name = self.params.name
        self.root_path = None
        self.checkpoint_path = None
        self.log_path = None

    def make_dirs(self):
        root_path = self.root + '/' + self.name
        if not os.path.isdir(root_path):
            os.makedirs(root_path)
        checkpoint_path = root_path + '/checkpoint'
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.root_path = root_path
        self.checkpoint_path = checkpoint_path + '/checkpoint'
        self.log_path = self.root_path + '/' + self.name + '.csv'

    def _check_log_exists(self):
        if os.path.isfile(self.log_path):
            return True
        else:
            return False

    def log_results(self, results):
        if self._check_log_exists():
            df = pd.read_csv(self.log_path, index_col=0)
            print(df)
            print(results)
            df.loc[len(df)] = np.array(results)
            df.to_csv(self.log_path)
        else:
            df = pd.DataFrame(np.array([results]),
                              columns=["Train Epoch Loss", "Train Epoch Acc", "Val Epoch Loss", "Val Epoch Acc",
                                       "Val Epoch Bleu"])
            df.to_csv(self.log_path)

    def save_model(self, epoch, model, optimizer):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, self.checkpoint_path)

    def save_params(self):
        with open(self.root_path + '/input_params.txt', 'w') as f:
            json.dump(self.params.__dict__, f, indent=2)


class TestLogger:
    def __init__(self, params):
        self.params = params
        self.root = self.params.location
        self.name = self.params.name
        self.test_name = self.test_name
        self.test_path = None
        self.test_log_path = None
        self.checkpoint_path = None
        self.target_examples = []
        self.pred_examples = []

    def make_dirs(self):
        test_path = self.root + '/test'
        checkpoint_path = self.root + '/checkpoint'
        if not os.path.isdir(test_path):
            os.makedirs(test_path)
        self.test_path = test_path
        self.test_log_path = test_path + '/' + self.name + '.csv'
        self.checkpoint_path = checkpoint_path

    def log_results(self, results):
        df = pd.DataFrame(np.array([results]),
                          columns=["Langs", "Test Loss", "Test Acc", "Test Bleu"])
        df.to_csv(self.test_log_path)

    def log_examples(self, target_batch, prediction_batch, tokenizer):
        det_target = str(detokenize(target_batch, tokenizer[1])[0])
        det_pred = str(detokenize(prediction_batch, tokenizer[1])[0])
        self.target_examples.append(det_target)
        self.pred_examples.append(det_pred)

    def dump_examples(self):
        with open(self.test_path + '/examples.txt', 'w') as f:
            for pred, target in zip(self.pred_examples, self.target_examples):
                f.write("Target: {} \n \n".format(target))
                f.write("Prediction: {} \n \n".format(pred))
                f.write("---------------------------------- \n \n")


def load_params(root_path):
    with open(root_path + '/input_params.txt', 'r') as f:
        param_dict = json.load(f)
    params = Loader(param_dict, check_custom=False)

    return params


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return model, optimizer, epoch
    else:
        return model
