from Transformer import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#from datasets import load_dataset
#from transformers import AutoTokenizer
import gc
import Dataset as ds
from Worker import Worker
import Utils
import Federated_training
import Metrics
from Server import Server

config = Utils.load_config("config.json")

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score

# Ensure nltk resources are downloaded
nltk.download('wordnet')

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


server = Server(config=config)

server.load_decrypt_model(file_path='sealed_models/worker0')

optimizer = optim.Adam(server.get_model().parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)#optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
dataset = ds.tokenized_dataset(name='wmt19', n_records_train = config['data_in_each_worker'], n_records_test = config['test_in_each_worker'], max_seq_length = config['max_seq_length'], train_offset = 0, test_offset = 0)

#base_seal_folder = "sealed_models"
paths = ["sealed_models/worker0","sealed_models/worker1","sealed_models/worker2"]
####################### Initialize workers based on the number specified in config dictionary
workers = Federated_training.initialize_workers(config)
main_server = Federated_training.initialize_server(config)

#paths = Federated_training.seal_store_models(workers)

workers = Federated_training.load_unseal_models(paths,workers)
workers[0].get_model().train()


# learning_rate = 0.01
# weight_decay = 0.0
# momentum = 0.9
# dampening = 0.0
# nesterov = False
# epsilon = 1e-8
# optimizer2 = optim.SGD(params=workers[1].get_model().parameters(), lr=learning_rate, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
# workers[1].optimizer = optimizer2
Federated_training.setup_optimizers(workers=workers, optimizer_name='sgd')
# Update the learning rate of the optimizer

# for param_group in workers[1].get_optimizer().param_groups:
#     param_group['lr'] = learning_rate
#     param_group['weight_decay'] = weight_decay
#     param_group['momentum'] = momentum
#     param_group['nesterov'] = nesterov
#     param_group['epsilon'] = epsilon
#     param_group['beta1'] = beta1
#     param_group['beta2'] = beta2

# Federated_training.train_workers(workers[1:2])

paths=paths[0:1]
#server = main_server.aggregate_models(paths, agg_method='fedAvg')  # FL aggregation happens here

workers[0].get_model().eval()
with torch.autograd.no_grad():#torch.no_grad():

      source_ids_validation = torch.tensor(dataset['source_ids_validation']).to(config['device'])
      target_ids_validation = torch.tensor(dataset['target_ids_validation']).to(config['device'])
      # Forward pass
      val_output = workers[0].get_model()(source_ids_validation, target_ids_validation)
      # Compute the loss
      val_loss = criterion(val_output.contiguous().view(-1, config['tgt_vocab_size']),target_ids_validation.contiguous().view(-1))
         # Print validation loss
      print(f"Validation Loss: {val_loss.item()}")
      generated_tokens = torch.argmax(val_output, dim=-1)

            # Convert token IDs to actual tokens using your vocabulary
            # Convert token IDs to actual tokens using the BART tokenizer
      generated_texts = ds.decode_tokens(generated_tokens)

# Example usage:
validation_predictions = generated_tokens.tolist() # List of generated translations
validation_ground_truths = target_ids_validation.tolist()   # List of ground truth translations

len(validation_ground_truths)
len(validation_predictions)

candidate_corpus = ds.decode_tokens(generated_tokens[0])
reference_corpus = ds.decode_tokens(target_ids_validation[0])

bleu_score = Metrics.calculate_bleu(candidate_corpus, reference_corpus)
print("BLEU score:", bleu_score)


candidate_corpuses = ds.decode_tokens(generated_tokens)
print(candidate_corpuses)




#Federated_training.send_global_model_to_clients(config, server=server['model'])  # send aggregated model to clients
#Federated_training.load_unseal_models(paths, workers)  # this function is equal to receieve model in client side
#Federated_training.train_workers(workers[1:2])



# Example usage:
predicted_texts = ["This is a test translation.", "Another translation example."]
reference_texts = ["This is a sample translation.", "Another example translation."]

rouge_scores = compute_rouge(predicted_texts, reference_texts)
#meteor_score = compute_meteor(predicted_texts, reference_texts)

print("ROUGE Scores:", rouge_scores)
#print("METEOR Score:", meteor_score)
