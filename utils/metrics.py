"""
Evaluation metrics for NMNT
"""

import torch
import nltk


def compute_bleu_sentence(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25)):
    """Compute Bleu Score for batches of data"""

    batch_size, n = ref.shape
    sum_bleu = 0.0
    for batch in range(0, batch_size):
        ref_sentence = ref[batch]
        hyp_sentence = hyp[batch]
        ref_sentence = ref_sentence[ref_sentence.nonzero()].reshape(-1).tolist()
        hyp_sentence = hyp_sentence[hyp_sentence.nonzero()].reshape(-1).tolist()
        sum_bleu += nltk.translate.bleu_score.sentence_bleu([ref_sentence], hyp_sentence, weights)

    bleu_score = sum_bleu / batch_size

    return bleu_score


if __name__ == "__main__":
    ref = torch.tensor([1, 2, 3, 4, 5]).tolist()
    hyp = torch.tensor([1, 3, 3, 4, 6, 7]).tolist()
    print(compute_bleu(ref, hyp))



