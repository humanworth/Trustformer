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
# Serialize and encrypt transformer model
#serialized_model = pickle.dumps(transformer_model)

class Worker:
    def __init__(self, dataset, config, transformer, name = 'worker1', provisioning_key = b'Sixteen byte key'):
        self.dataset = dataset # this is a dictionary of {"","","",""}
        self.key = provisioning_key
        self.name = name
        self.optimizer = optim.SGD(transformer.parameters(), lr=0.001, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
            #optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9))#optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.transformer = transformer
        self.history = {}
        self.config = config
        
    def start_training(self):
        self.transformer.train()
        self.history['iteration'] = []
        self.history['batch'] = []
        self.history['loss'] = []
        self.history['accuracy'] = []
        for iteration in range(len(self.dataset['source_ids'])//self.config['batch_size']):   
            for batch in range(self.config['batch_size']):
                src_data = torch.tensor(self.dataset['source_ids'][iteration*self.config['batch_size']:(iteration+1)*self.config['batch_size']]).to(self.config['device'])
                tgt_data = torch.tensor(self.dataset['target_ids'][iteration*self.config['batch_size']:(iteration+1)*self.config['batch_size']]).to(self.config['device'])
                self.optimizer.zero_grad()
                self.transformer.to(self.config['device'])
                output = self.transformer(src_data, tgt_data)#transformer(src_data, tgt_data[:,:-1])
                # Loss Calculation
                #loss = loss_function(output, tgt_data)
                loss_1 = self.criterion(output.contiguous().view(-1, self.config['tgt_vocab_size']), tgt_data.contiguous().view(-1))#tgt_data[:, 1:].contiguous().view(-1))
                loss_1.to(self.config['device'])
                loss_1.backward()
                self.optimizer.step()
                print(f"iteration: {iteration+1}, batch: {batch+1} , Loss: {loss_1.item()} of transformer {self.name}")
                self.history['iteration'] = self.history['iteration'].append(iteration + 1)
                self.history['batch'] = self.history['batch'].append(batch + 1)
                self.history['loss'] = self.history['loss'].append(loss_1.item())
                # self.history['accuracy'] = self.history['accuracy'].append()
                del src_data
                del tgt_data
                torch.cuda.empty_cache()
        return self.transformer
    
    def evaluate_model(self):
        self.transformer.eval()
        with torch.autograd.no_grad():#torch.no_grad():
            source_ids_validation = torch.tensor(self.dataset['source_ids_validation']).to(self.config['device'])
            target_ids_validation = torch.tensor(self.dataset['target_ids_validation']).to(self.config['device'])
            # Forward pass
            val_output = self.transformer(source_ids_validation, target_ids_validation)
            # Compute the loss
            val_loss = self.criterion(val_output.contiguous().view(-1, self.config['tgt_vocab_size']),target_ids_validation.contiguous().view(-1))

            # Print validation loss
            print(f"Validation Loss: {val_loss.item()}")

            generated_tokens = torch.argmax(val_output, dim=-1)

            # Convert token IDs to actual tokens using your vocabulary
            # Convert token IDs to actual tokens using the BART tokenizer
            generated_texts = ds.decode_tokens(generated_tokens)
        return generated_texts
    
    def get_model(self):
        return self.transformer
    def get_optimizer(self):
        return self.optimizer
    def set_parameters(self, new_transformer):
        for param_new, param1 in zip(new_transformer.parameters(), self.transformer.parameters()):
            # Check if the parameter is trainable
            if param_new.requires_grad:
                # Average the parameter values
                param1.data.copy_(param_new.data)
       
    
    def load_decrypt_model(self, file_path):
        loaded_transformer = sgx.load_model(file_path=file_path, key=self.key)
        self.transformer = loaded_transformer
        return self

    def encrypt_store_model(self):
        print('SGX encrypting and storing the transformer of : ', self.name)
        directory = "sealed_models"
        if not os.path.exists(directory):
            os.makedirs(directory)        
        file_path = "sealed_models/{}".format(self.name)
        sgx.store_model(self.key, self.transformer, filename=file_path)
        return file_path
    