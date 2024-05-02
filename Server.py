# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:01:47 2024

@author: aliab
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:33:00 2024

@author: aliab
"""
import SGXEmulator as sgx
import pickle
import torch
import Dataset as ds
import torch.optim as optim
import torch.nn as nn
import torch
import os
from Transformer import Transformer


# Serialize and encrypt transformer model
# serialized_model = pickle.dumps(transformer_model)


class Server:
    def __init__(self, config, name='server', provisioning_key=b'Sixteen byte key'):
        self.key = provisioning_key
        self.name = name
        self.config = config
        self.server_model = Transformer(src_vocab_size=self.config['src_vocab_size'],
                                        tgt_vocab_size=self.config['tgt_vocab_size']
                                        , d_model=self.config['d_model'], num_heads=self.config['num_heads']
                                        , num_layers=self.config['num_layers'], d_ff=self.config['d_ff']
                                        , max_seq_length=self.config['max_seq_length'], dropout=self.config['dropout']
                                        , transformer_id=100)
        self.server_model.to(self.config['device'])
    def aggregate_models(self, transformers, agg_method='fedAvg'):
        if agg_method == 'fedAvg':
            differences = []
            for transformer in transformers:
                loaded_transformer = sgx.load_model(file_path=transformer, key=self.key)
                loaded_transformer.to(self.config['device'])
                ###### Compute difference between workers and server model
                differences.append(self.calculate_difference(loaded_transformer))
            ###### Average the difference
            average = self.average_differences(differences=differences)
            ###### subtract the average from server
            self.set_new_weights(average)

            file_path = self.encrypt_store_model(name='server')
            return {"model": self, "file_path": file_path}

    def set_new_weights(self, average):
        for avg, server in zip(average.parameters(), self.server_model.parameters()):
            # Check if the parameter is trainable.
            if server.requires_grad:
                # subtract all transformer parameters
                server.data.copy_(server.data - avg.data)

    def average_differences(self, differences):
        placeholder = Transformer(src_vocab_size=self.config['src_vocab_size'],
                                  tgt_vocab_size=self.config['tgt_vocab_size']
                                  , d_model=self.config['d_model'], num_heads=self.config['num_heads']
                                  , num_layers=self.config['num_layers'], d_ff=self.config['d_ff']
                                  , max_seq_length=self.config['max_seq_length'], dropout=self.config['dropout']
                                  , transformer_id=100)
        placeholder.to(self.config['device'])
        for param in placeholder.parameters():
            param.data.fill_(0)
        for difference in differences:
            for holder, param in zip(placeholder.parameters(), difference.parameters()):
                # Check if the parameter is trainable.
                if param.requires_grad:
                    # sum all transformer parameters
                    holder.data.copy_(param.data + holder.data)

        for holder in placeholder.parameters():
            # Check if the parameter is trainable.
            if holder.requires_grad:
                # Average the parameter values
                holder.data.copy_(holder.data / len(differences))
        return placeholder

    def get_model(self):
        return self.server_model

    def load_decrypt_model(self, file_path):
        self.transformer = sgx.load_model(file_path=file_path, key=self.key)
        return self

    # def encrypt_store_model(self):
    #     print('SGX encrypting and storing the transformer of : ', self.name)
    #     directory = "sealed_models"
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)        
    #     file_path = "sealed_models/{}".format(self.name)
    #     sgx.store_model(self.key, self.server_model, filename=file_path)
    #     return file_path

    def evaluate_model(self):
        self.transformer.eval()
        with torch.autograd.no_grad():  # torch.no_grad():
            source_ids_validation = torch.tensor(self.dataset['source_ids_validation']).to(self.config['device'])
            target_ids_validation = torch.tensor(self.dataset['target_ids_validation']).to(self.config['device'])
            # Forward pass
            val_output = self.transformer(source_ids_validation, target_ids_validation)
            # Compute the loss
            val_loss = self.criterion(val_output.contiguous().view(-1, self.config['tgt_vocab_size']),
                                      target_ids_validation.contiguous().view(-1))

            # Print validation loss
            print(f"Validation Loss: {val_loss.item()}")

            generated_tokens = torch.argmax(val_output, dim=-1)

            # Convert token IDs to actual tokens using your vocabulary
            # Convert token IDs to actual tokens using the BART tokenizer
            generated_texts = ds.decode_tokens(generated_tokens)
        return generated_texts

    def encrypt_store_model(self, name):
        print('SGX encrypting and storing the transformer of : ', name)
        directory = "sealed_models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = "sealed_models/{}".format(name)
        sgx.store_model(self.key, self.server_model, filename=file_path)
        return file_path

    def calculate_difference(self, loaded_transformer):
        output_t = Transformer(src_vocab_size=self.config['src_vocab_size'],
                             tgt_vocab_size=self.config['tgt_vocab_size']
                             , d_model=self.config['d_model'], num_heads=self.config['num_heads']
                             , num_layers=self.config['num_layers'], d_ff=self.config['d_ff']
                             , max_seq_length=self.config['max_seq_length'], dropout=self.config['dropout']
                             , transformer_id=100)
        output_t.to(self.config['device'])
        for param in output_t.parameters():
            param.data.fill_(0)
        for server, param, output in zip(self.server_model.parameters(), loaded_transformer.parameters(),output_t.parameters()):
            # Check if the parameter is trainable.
            if server.requires_grad:
                # sum all transformer parameters
                output.data.copy_(server.data - param.data)
        return output_t
