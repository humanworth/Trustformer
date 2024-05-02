# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:43:45 2024

@author: aliab
"""
import torch

# from torchtext.data.metrics import bleu_score


def calculate_accuracy(predictions, ground_truths):
    correct_translations = 0
    total_sentences = len(predictions)
    
    for pred, truth in zip(predictions, ground_truths):
        if pred == truth:
            correct_translations += 1
    
    accuracy = correct_translations / total_sentences
    return accuracy


# def calculate_bleu(candidate_tensor, reference_tensor):
#     bleu = bleu_score(candidate_tensor, reference_tensor,max_n=3,weights=[0.001]*3)
#     return bleu