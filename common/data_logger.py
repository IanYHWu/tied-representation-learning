"""
Writing useful things to disk
"""

import os
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import json
from tokenizers import Tokenizer

from hyperparams.loader import Loader
from common.preprocess import detokenize
from common.utils import mask_after_stop


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
        Path(root_path).mkdir(parents=True, exist_ok=True)
        checkpoint_path = root_path + '/checkpoint'
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
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
                              columns=["Train Epoch Loss", "Train Epoch Loss Aux",
                              "Train Epoch Acc", "Val Epoch Loss",
                              "Val Epoch Acc", "Val Bleu", "Test Bleu"])
            df.to_csv(self.log_path)

    def save_model(self, epoch, model, optimizer, scheduler=None):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(save_dict, self.checkpoint_path)

    def save_params(self):
        with open(self.root_path + '/input_params.txt', 'w') as f:
            json.dump(self.params.__dict__, f, indent=2)


class TestLogger:
    def __init__(self, params):
        self.params = params
        self.root = self.params.location
        self.name = self.params.name
        self.test_name = self.params.test_name
        self.test_path = None
        self.test_log_path = None
        self.checkpoint_path = None
        self.root_path = None
        self.input_examples = []
        self.target_examples = []
        self.pred_examples = []

    def make_dirs(self):
        root_path = self.root + '/' + self.name
        test_path = root_path + '/test'
        checkpoint_path = root_path + '/checkpoint'
        Path(test_path).mkdir(parents=True, exist_ok=True)
        self.test_path = test_path
        self.root_path = root_path
        self.test_log_path = test_path + '/' + self.test_name + '.csv'
        self.checkpoint_path = checkpoint_path + '/checkpoint'

    def log_results(self, results, columns=None):
        if columns is None:
            columns = ["direction", "test_acc", "test_bleu"]
        assert len(results) == len(columns)
        results = [r if isinstance(r, list) else [r] for r in results]
        df = pd.DataFrame({c:r for c,r in zip(columns, results)})
        df.to_csv(self.test_log_path, index=False)

    def log_examples(self, input_batch, target_batch, prediction_batch, tokenizer):
        prediction_batch = mask_after_stop(prediction_batch, stop_token=2)
        if isinstance(tokenizer, list):
            inp_tokenizer = tokenizer[0]
            out_tokenizer = tokenizer[1]
        else:
            inp_tokenizer = tokenizer
            out_tokenizer = tokenizer
        det_input = str(detokenize(input_batch, inp_tokenizer, batch=True)[0])
        det_target = str(detokenize(target_batch, out_tokenizer, batch=True)[0])
        det_pred = str(detokenize(prediction_batch, out_tokenizer, batch=True)[0])

        self.target_examples.append(det_target)
        self.pred_examples.append(det_pred)
        self.input_examples.append(det_input)

    def dump_examples(self):
        with open(self.test_path + '/' + self.test_name + '_examples.txt', 'w') as f:
            for inp, pred, target in zip(self.input_examples, self.pred_examples, self.target_examples):
                f.write("Input: {} \n \n".format(inp))
                f.write("Target: {} \n \n".format(target))
                f.write("Prediction: {} \n \n".format(pred))
                f.write("---------------------------------- \n \n")


def load_params(root_path):
    with open(root_path + '/input_params.txt', 'r') as f:
        param_dict = json.load(f)
    params = Loader(param_dict, check_custom=False)

    return params


def load_checkpoint(path, device, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, epoch, scheduler

