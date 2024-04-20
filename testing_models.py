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
workers[1].get_model().train()


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

paths=paths[1:2]
server = main_server.aggregate_models(paths, agg_method='fedAvg')  # FL aggregation happens here

workers[1].get_model().eval()
with torch.autograd.no_grad():#torch.no_grad():

      source_ids_validation = torch.tensor(dataset['source_ids_validation']).to(config['device'])
      target_ids_validation = torch.tensor(dataset['target_ids_validation']).to(config['device'])
      # Forward pass
      val_output = workers[1].get_model()(source_ids_validation, target_ids_validation)
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




Federated_training.send_global_model_to_clients(config, server=server['model'])  # send aggregated model to clients
Federated_training.load_unseal_models(paths, workers)  # this function is equal to receieve model in client side
Federated_training.train_workers(workers[1:2])