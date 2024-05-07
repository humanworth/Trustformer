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
import torch
from torch.cuda.amp import GradScaler, autocast
import gc
from NoamLR import NoamLR
import Metrics
# Serialize and encrypt transformer model
# serialized_model = pickle.dumps(transformer_model)

class Worker:
    def __init__(self, dataset, config, transformer, name='worker1', provisioning_key=b'Sixteen byte key'):
        self.dataset = dataset  # this is a dictionary of {"","","",""}
        self.key = provisioning_key
        self.name = name
        # self.optimizer = optim.SGD(transformer.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
        #                            dampening=config['dampening'], weight_decay=config['weight_decay'],
        #                            nesterov=config['nesterov'])
        self.optimizer =optim.Adam(transformer.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-3)
        self.criterion = nn.CrossEntropyLoss(ignore_index=1)
        self.transformer = transformer
        self.history = {}
        self.config = config
        # self.transformer.half()

    # def start_training(self):
    #     # Set the model to training mode and move it to the device
    #     self.transformer.train()
    #     self.transformer.to(self.config['device'])
    #
    #     # Initialize history tracking
    #     self.history = {
    #         'iteration': [],
    #         'batch': [],
    #         'loss': [],
    #         'accuracy': []
    #     }
    #
    #     # Create a GradScaler for mixed precision training
    #     scaler = GradScaler()
    #
    #     # Loop through the dataset in batches
    #     for iteration in range(len(self.dataset['source_ids']) // self.config['batch_size']):
    #         for batch in range(self.config['batch_epoch']):
    #             # Prepare batch data
    #             src_data = torch.tensor(
    #                 self.dataset['source_ids'][
    #                 iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']]
    #             ).to(self.config['device'])
    #
    #             tgt_data = torch.tensor(
    #                 self.dataset['target_ids'][
    #                 iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']]
    #             ).to(self.config['device'])
    #
    #             # Zero out gradients
    #             self.optimizer.zero_grad()
    #
    #             # Mixed precision training with autocast
    #             with autocast():
    #                 output = self.transformer(src_data, tgt_data)
    #                 loss_1 = self.criterion(
    #                     output.contiguous().view(-1, self.config['tgt_vocab_size']),
    #                     tgt_data.contiguous().view(-1)
    #                 )
    #
    #             # Scale the loss for mixed precision training
    #             # torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
    #
    #             scaler.scale(loss_1).backward()
    #             scaler.unscale_(self.optimizer)
    #             torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
    #
    #             try:
    #                 scaler.step(self.optimizer)
    #             finally:
    #                 scaler.update()
    #
    #
    #             # Logging information
    #             print(
    #                 f"iteration: {iteration + 1}, batch_epoch: {batch + 1}, Loss: {loss_1.item()} of transformer {self.name}")
    #
    #             # Track history
    #             self.history['iteration'].append(iteration + 1)
    #             self.history['batch'].append(batch + 1)
    #             self.history['loss'].append(loss_1.item())
    #
    #             # Clear memory by deleting references
    #             src_data.to('cpu')
    #             tgt_data.to('cpu')
    #             del src_data, tgt_data, loss_1, output
    #             torch.cuda.empty_cache()
    #
    #             # Trigger garbage collection to reclaim memory
    #         gc.collect()
    #     torch.cuda.empty_cache()
    #     return self.transformer
    def start_training(self):
        # Set the model to training mode and move it to the device
        self.transformer.train()
        self.transformer.to(self.config['device'])

        # Initialize history tracking
        self.history = {
            'iteration': [],
            'batch': [],
            'loss': [],
            'accuracy': []  # Ensure accuracy is computed if needed.
        }
        # scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        # scheduler = scheduler = NoamLR(model_size=512, warmup_steps=4000)

        # Loop through the dataset in batches
        for iteration in range(len(self.dataset['source_ids']) // self.config['batch_size']):
            for batch in range(self.config['batch_epoch']):
                # Prepare batch data directly on the device, minimizing memory copy operations
                src_data = torch.tensor(
                    self.dataset['source_ids'][
                    iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                    device=self.config['device'], dtype=torch.long)  # Ensure correct dtype to match model expectations

                tgt_data = torch.tensor(
                    self.dataset['target_ids'][
                    iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                    device=self.config['device'], dtype=torch.long)

                # Zero out gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.transformer(src_data, tgt_data)
                loss = self.criterion(
                    output.contiguous().view(-1, self.config['tgt_vocab_size']),
                    tgt_data.contiguous().view(-1)
                )

                # Backward pass
                loss.backward()

                # Update parameters
                self.optimizer.step()

                # Logging information
                print(f"iteration: {iteration + 1}, batch_epoch: {batch + 1}, Loss: {loss.item()} of transformer {self.name}")
                # print("Optimizer state keys after update:", self.optimizer.state_dict()['state'].keys())

                # Track history
                self.history['iteration'].append(iteration + 1)
                self.history['batch'].append(batch + 1)
                self.history['loss'].append(loss.item())

                # Optional: Clear cache periodically to free unused memory

                # scheduler.step()
                # Manually clear variables to ensure they are deleted from memory
                del src_data, tgt_data, output, loss
                torch.cuda.empty_cache()
                # Trigger garbage collection to reclaim memory
                gc.collect()

        self.transformer.to('cpu')
        torch.cuda.empty_cache()
        return self.transformer


    def evaluate_model(self):
        self.transformer.eval()
        self.transformer.to(self.config['device'])
        text_results = []
        loss_results = []
        metrics = {'bert_score':[],'bleu_score':[],'rouge_scores':[], 'meteor_score':[]}
        with (torch.autograd.no_grad()):  # torch.no_grad():
            for iteration in range(len(self.dataset['source_ids_validation']) // self.config['batch_size']):
                source_ids_validation = torch.tensor(
                    self.dataset['source_ids_validation'][
                    iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                    device=self.config['device'], dtype=torch.long)  # Ensure correct dtype to match model expectations

                target_ids_validation = torch.tensor(
                    self.dataset['target_ids_validation'][
                    iteration * self.config['batch_size']:(iteration + 1) * self.config['batch_size']],
                    device=self.config['device'], dtype=torch.long)

                # source_ids_validation = torch.tensor(self.dataset['source_ids_validation']).to(self.config['device'])
                # target_ids_validation = torch.tensor(self.dataset['target_ids_validation']).to(self.config['device'])

                # Forward pass
                val_output = self.transformer(source_ids_validation, target_ids_validation)
                # Compute the loss
                val_loss = self.criterion(val_output.contiguous().view(-1, self.config['tgt_vocab_size']),
                                      target_ids_validation.contiguous().view(-1))

                # Print validation loss
                print(f"Validation Loss: {val_loss.item()}")
                loss_results.append(val_loss.item())
                generated_tokens = torch.argmax(val_output, dim=-1)

                # Convert token IDs to actual tokens using the BART tokenizer
                # generated_texts = ds.decode_tokens(generated_tokens)
                # text_results.append(generated_texts)
                candidate_corpus = ds.decode_tokens(generated_tokens)
                text_results.append(candidate_corpus)
                reference_corpus = ds.decode_tokens(target_ids_validation)

                # candidate_corpuses = ds.decode_tokens(generated_tokens)
                print(candidate_corpus)

                P, R, F1 = Metrics.score(candidate_corpus, reference_corpus, lang="en")
                print(f"BERTScore Precision: {P.mean()}, Recall: {R.mean()}, F1 Score: {F1.mean()}")
                # metrics['bert_score'] = {"P": P.mean(), "R":R.mean(), "F1 score": F1.mean()}
                # for key in metrics['bert_score']:
                metrics['bert_score'].append({"P": P.mean(), "R":R.mean(), "F1 score": F1.mean()})
                bleu_score = Metrics.sentence_bleu(reference_corpus, candidate_corpus)
                print(f"BLEU Score: {bleu_score}")
                metrics['bleu_score'].append(bleu_score)
                rouge_scores = Metrics.compute_rouge(candidate_corpus, reference_corpus)
                print("ROUGE Scores:", rouge_scores)
                metrics['rouge_scores'].append(rouge_scores)
                meteor_score_1 = Metrics.compute_meteor([candidate_corpus], [reference_corpus])
                print("METEOR Score:", meteor_score_1)
                metrics['meteor_score'].append(meteor_score_1)

                del source_ids_validation, target_ids_validation, val_output, val_loss,generated_tokens,candidate_corpus,  P, R, F1, bleu_score, rouge_scores, meteor_score_1
                torch.cuda.empty_cache()
                # Trigger garbage collection to reclaim memory
                gc.collect()

            self.transformer.to('cpu')
            torch.cuda.empty_cache()
        return text_results, loss_results , metrics

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

    def set_optimizer(self, optimizer, optimizer_name='sgd'):
        if optimizer_name == 'sgd':
            optimizer2 = optim.SGD(params=self.transformer.parameters(), lr=self.config["learning_rate"],
                                   momentum=self.config["momentum"],
                                   dampening=self.config["dampening"], weight_decay=self.config["weight_decay"],
                                   nesterov=self.config["nesterov"])
            # scheduler = optim.lr_scheduler.StepLR(optimizer2, step_size=3, gamma=0.1)
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.95)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)
            # scheduler.step()

        if optimizer_name == 'adam':
            # scheduler = optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.1)
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.95)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)
            # scheduler.step()
            self.optimizer = optimizer

    def encrypt_store_model(self):
        print('SGX encrypting and storing the transformer of : ', self.name)
        directory = "sealed_models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = "sealed_models/{}".format(self.name)
        sgx.store_model(self.key, self.transformer, filename=file_path)
        return file_path

    def set_model(self, newTransformer):
        self.transformer = newTransformer
