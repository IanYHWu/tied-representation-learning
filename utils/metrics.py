"""
Evaluation metrics for NMNT
"""

import nltk


def compute_bleu(ref, hyp, weights=(0.5, 0.5)):
    """Compute Bleu Score"""
    bleu_score = nltk.translate.bleu_score.sentence_bleu([ref], [hyp], weights)

    return bleu_score


