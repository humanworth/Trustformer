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
from bert_score import score

config = Utils.load_config("config.json")

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
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

server.load_decrypt_model(file_path='sealed_models/server')

optimizer = optim.Adam(server.get_model().parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)#optimizer
criterion = nn.CrossEntropyLoss(ignore_index=1)
dataset = ds.tokenized_dataset(name='wmt19', n_records_train = config['data_in_each_worker'], n_records_test = config['test_in_each_worker'], max_seq_length = config['max_seq_length'], train_offset = 0, test_offset = 0)

#base_seal_folder = "sealed_models"
paths = ["sealed_models/worker0","sealed_models/worker1","sealed_models/worker2"]
####################### Initialize workers based on the number specified in config dictionary
workers = Federated_training.initialize_workers(config)
main_server = Federated_training.initialize_server(config)
server = main_server.aggregate_models(paths, agg_method='Avg')  # FL aggregation happens here


# paths = Federated_training.seal_store_models(workers)
# Federated_training.send_global_model_to_clients(config, server=server['model'])  # send aggregated model to clients
# Federated_training.load_unseal_models(paths, workers)  # this function is equal to receieve model in client side
# workers[0].set_model(server['model'].get_model())
Federated_training.train_workers(workers)
new_optimizer = main_server.aggregate_optimizers([worker.optimizer for worker in workers])
Federated_training.setup_optimizers(new_optimizer, workers=workers, optimizer_name='adam')
# paths = Federated_training.seal_store_models(workers[0:1])

# workers = Federated_training.load_unseal_models(paths,workers)
# workers[0].get_model().train()



# Federated_training.train_workers(workers[0:1])

# paths=paths[0:1]
workers[0].get_model().to(config['device'])
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

candidate_corpus = ds.decode_tokens(generated_tokens)
reference_corpus = ds.decode_tokens(target_ids_validation)

candidate_corpuses = ds.decode_tokens(generated_tokens)
print(candidate_corpuses)






P, R, F1 = score(candidate_corpus, reference_corpus, lang="en")
print(f"BERTScore Precision: {P.mean()}, Recall: {R.mean()}, F1 Score: {F1.mean()}")
bleu_score = sentence_bleu(reference_corpus, candidate_corpus)
print(f"BLEU Score: {bleu_score}")
rouge_scores = compute_rouge(candidate_corpus, reference_corpus)
print("ROUGE Scores:", rouge_scores)
meteor_score = compute_meteor([candidate_corpus], [reference_corpus])
print("METEOR Score:", meteor_score)
