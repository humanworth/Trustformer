# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:43:45 2024

@author: aliab
"""
import torch
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
from bert_score import score

from nltk.translate.bleu_score import sentence_bleu

# Ensure nltk resources are downloaded
nltk.download('wordnet')


# from torchtext.data.metrics import bleu_score


def calculate_accuracy(predictions, ground_truths):
    correct_translations = 0
    total_sentences = len(predictions)

    for pred, truth in zip(predictions, ground_truths):
        if pred == truth:
            correct_translations += 1

    accuracy = correct_translations / total_sentences
    return accuracy


def compute_rouge(predicted_texts, reference_texts):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize accumulators for each ROUGE metric
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for pred, ref in zip(predicted_texts, reference_texts):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    # Compute average scores
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return {
        'ROUGE-1': avg_rouge1,
        'ROUGE-2': avg_rouge2,
        'ROUGE-L': avg_rougeL,
    }


def compute_meteor(predicted_texts, reference_texts):
    # Compute METEOR scores for each pair
    meteor_scores = [meteor_score([ref], pred) for pred, ref in zip(predicted_texts, reference_texts)]

    # Compute average METEOR score
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return avg_meteor


def score_bert(reference, candidate):
    return score(candidate, reference, lang="en")
