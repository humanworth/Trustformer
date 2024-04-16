# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:48:15 2024

@author: aliab
"""

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


config = Utils.load_config("config.json")
 
#base_seal_folder = "sealed_models"
paths = ["sealed_models/worker0","sealed_models/worker1","sealed_models/worker2"]
####################### Initialize workers based on the number specified in config dictionary
workers = Federated_training.initialize_workers(config)
main_server = Federated_training.initialize_server(config)

#skip as they are already trained and stored in bin file worker0,worker1,worker2

for epoch in range(config['n_epochs']):
    print("start training for epoch {}".format(epoch))
    Federated_training.train_workers(workers)
    paths = Federated_training.seal_store_models(workers)
    server = main_server.aggregate_models(paths, agg_method='fedAvg') # FL aggregation happens here
    Federated_training.send_global_model_to_clients(config,server=server['model']) #send aggregated model to clients
    Federated_training.load_unseal_models(paths, workers) # this function is equal to receieve model in client side

#load_workers[0].start_training()



















######################## Juts for test
# optimizer = optim.Adam(server['model'].get_model().parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)#optimizer
# criterion = nn.CrossEntropyLoss(ignore_index=0)
# dataset = ds.tokenized_dataset(name='wmt19', n_records_train = config['data_in_each_worker'], n_records_test = config['test_in_each_worker'], max_seq_length = config['max_seq_length'], train_offset = 0, test_offset = 0)
#
# server['model'].get_model().eval()
# with torch.autograd.no_grad():#torch.no_grad():
#
#       source_ids_validation = torch.tensor(dataset['source_ids_validation']).to(config['device'])
#       target_ids_validation = torch.tensor(dataset['target_ids_validation']).to(config['device'])
#       # Forward pass
#       val_output = server['model'].get_model()(source_ids_validation, target_ids_validation)
#       # Compute the loss
#       val_loss = criterion(val_output.contiguous().view(-1, config['tgt_vocab_size']),target_ids_validation.contiguous().view(-1))
#          # Print validation loss
#       print(f"Validation Loss: {val_loss.item()}")
#       generated_tokens = torch.argmax(val_output, dim=-1)
#
#             # Convert token IDs to actual tokens using your vocabulary
#             # Convert token IDs to actual tokens using the BART tokenizer
#       generated_texts = ds.decode_tokens(generated_tokens)
#
# # Example usage:
# validation_predictions = generated_tokens.tolist() # List of generated translations
# validation_ground_truths = target_ids_validation.tolist()   # List of ground truth translations
#
# len(validation_ground_truths)
# len(validation_predictions)
#
# candidate_corpus = ds.decode_tokens(generated_tokens[0])
# reference_corpus = ds.decode_tokens(target_ids_validation[5])
#
# bleu_score = Metrics.calculate_bleu(candidate_corpus, reference_corpus)
# print("BLEU score:", bleu_score)

    
##############################################################################



# sum_of_all_data_points = len(dataset[0]['source']) + len(dataset[1]['source']) + len(dataset[2]['source'])


# Average the trainable parameters
# for param_new, param1, param2, param3 in zip(transformer_center.parameters(), transformer_1.parameters(), transformer_2.parameters(), transformer_3.parameters()):
#     # Check if the parameter is trainable.
#     if param_new.requires_grad:
#        # print("Grad update")
#         # Average the parameter values
#         param_new.data.copy_(((len(dataset[0]['source']) * param1.data)/sum_of_all_data_points + 
#                               (len(dataset[1]['source'])* param2.data)/sum_of_all_data_points + 
#                               (len(dataset[2]['source']) * param3.data)/sum_of_all_data_points))

########################################################################



# adjust_parameters(transformer_1,operator="lower")
# adjust_parameters(transformer_2,operator="lower")
# adjust_parameters(transformer_3,operator="lower")

# for name, param in server[""].named_parameters():
#      #print(f"Name: {name}")
#      print(f"Params: {param.requires_grad}")
