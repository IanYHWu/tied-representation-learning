"""
Evaluation metrics for NMNT
"""

import torch
from collections import Counter
from allennlp.training.util import ngrams
from allennlp.training.util import get_valid_tokens_mask
import math
import numpy as np


class BLEU:
    def __init__(self, ngram_weights=(0.25, 0.25, 0.25, 0.25), exclude_indices=None):
        self._ngram_weights = ngram_weights
        self._precision_matches = Counter()
        self._precision_totals = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0
        self._exclude_indices = exclude_indices

    def reset(self):
        self._precision_matches = Counter()
        self._precision_totals = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0

    def set_excluded_indices(self, exclude_list):
        exclude_set = set()
        for i in exclude_list:
            exclude_set.add(i)

        self._exclude_indices = exclude_set

    def _get_brevity_penalty(self):
        if self._prediction_lengths > self._reference_lengths:
            return 1.0
        if self._reference_lengths == 0 or self._prediction_lengths == 0:
            return 0.0
        return math.exp(1.0 - self._reference_lengths / self._prediction_lengths)

    def _get_modified_precision_counts(self, predicted_tokens, reference_tokens, ngram_size):
        clipped_matches = 0
        total_predicted = 0
        for predicted_row, reference_row in zip(predicted_tokens, reference_tokens):
            predicted_ngram_counts = ngrams(predicted_row, ngram_size, self._exclude_indices)
            reference_ngram_counts = ngrams(reference_row, ngram_size, self._exclude_indices)
            for ngram, count in predicted_ngram_counts.items():
                clipped_matches += min(count, reference_ngram_counts[ngram])
                total_predicted += count
        return clipped_matches, total_predicted

    def __call__(self, predictions, gold_targets):
        for ngram_size, _ in enumerate(self._ngram_weights, start=1):
            precision_matches, precision_totals = self._get_modified_precision_counts(
                predictions, gold_targets, ngram_size)
            self._precision_matches[ngram_size] += precision_matches
            self._precision_totals[ngram_size] += precision_totals

        if not self._exclude_indices:
            _prediction_lengths = predictions.size(0) * predictions.size(1)
            _reference_lengths = gold_targets.size(0) * gold_targets.size(1)

        else:
            valid_predictions_mask = get_valid_tokens_mask(predictions, self._exclude_indices)
            valid_gold_targets_mask = get_valid_tokens_mask(gold_targets, self._exclude_indices)
            _prediction_lengths = valid_predictions_mask.sum().item()
            _reference_lengths = valid_gold_targets_mask.sum().item()

        self._prediction_lengths += _prediction_lengths
        self._reference_lengths += _reference_lengths

    def get_metric(self, reset=False):
        brevity_penalty = self._get_brevity_penalty()
        ngram_scores = (
            weight * (math.log(self._precision_matches[n] + 1e-13) - math.log(self._precision_totals[n] + 1e-13))
            for n, weight in enumerate(self._ngram_weights, start=1))
        bleu = brevity_penalty * math.exp(sum(ngram_scores))
        if reset:
            self.reset()

        return bleu
