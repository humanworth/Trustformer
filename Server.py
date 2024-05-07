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
        self.server_model.half()
        # self.server_model.to(self.config['device'])
    def aggregate_optimizers(self, optimizers):
        # for key in optimizer_states[0]['state'].keys():
        #     # Aggregate 'exp_avg' and 'exp_avg_sq'
        #     exp_avgs = torch.mean(torch.stack([client.state['state'][key]['exp_avg']
        #                                        for client in optimizers]), dim=0)
        #     exp_avg_sqs = torch.mean(
        #         torch.stack([client.state['state'][key+1]['exp_avg_sq'] for client in optimizers]),
        #         dim=0)
        #
        #     global_optimizer.state['state'][key+1] = {
        #         'step': 0,  # We reset the step counter
        #         'exp_avg': exp_avgs,
        #         'exp_avg_sq': exp_avg_sqs
        #     }

        for param_group in optimizers[0].param_groups:
            for param in param_group['params']:
                # param_group['lr'] = param_group['lr'] * 0.9
                if param in optimizers[0].state:
                    # Initialize tensors to store aggregated values
                    exp_avg_agg = None
                    exp_avg_sq_agg = None

                    # Collect all 'exp_avg' and 'exp_avg_sq' across all optimizers for the current param
                    exp_avgs = [opt.state[param]['exp_avg'] for opt in optimizers if param in opt.state]
                    exp_avg_sqs = [opt.state[param]['exp_avg_sq'] for opt in optimizers if param in opt.state]

                    # Stack and average the collected 'exp_avg' and 'exp_avg_sq'
                    if exp_avgs and exp_avg_sqs:
                        exp_avg_agg = torch.mean(torch.stack(exp_avgs), dim=0)
                        exp_avg_sq_agg = torch.mean(torch.stack(exp_avg_sqs), dim=0)

                    # Optionally set the aggregated values back to each optimizer (if needed)
                    for opt in optimizers:
                        if param in opt.state:
                            opt.state[param]['exp_avg'].copy_(exp_avg_agg)
                            opt.state[param]['exp_avg_sq'].copy_(exp_avg_sq_agg)
                            # opt.state[param]['step'] =  [torch.tensor(0)]
        return optimizers
    def aggregate_models(self, transformers, agg_method='fedAvg'):
        loaded_transformers = []
        for transformer in transformers:
            loaded_transformers.append(sgx.load_model(file_path=transformer, key=self.key))
        if agg_method == 'fedAvg':
            differences = []
            for transformer in loaded_transformers:
                # loaded_transformer.to(self.config['device'])
                ###### Compute difference between workers and server model
                differences.append(self.calculate_difference(transformer))
            ###### Average the difference
            average = self.average_differences(differences=differences)
            ###### subtract the average from server
            self.set_new_weights(average)
            # torch.cuda.empty_cache()
        if agg_method == 'Avg':
            """
            Aggregates the models from each client by averaging their state dicts.
            """
            global_dict = self.server_model.state_dict()

            # Summing all the models' state_dicts
            for key in global_dict.keys():
                global_dict[key] = torch.stack(
                    [loaded_transformers[i].state_dict()[key].float() for i in range(len(loaded_transformers))], 0).mean(0)

            # Update the global model
            self.server_model.load_state_dict(global_dict)


        file_path = self.encrypt_store_model(name='server')
        return {"model": self, "file_path": file_path}

    def set_new_weights(self, average):
        for avg, server in zip(average.parameters(), self.server_model.parameters()):
            # Check if the parameter is trainable.
            if server.requires_grad:
                # subtract all transformer parameters
                server.data.copy_(server.data - avg.data)
        # nn.utils.clip_grad_norm_(average.parameters(), max_norm=1.0)

    def check_inf_in_model(model):
        inf_parameters = {}
        for name, param in model.named_parameters():
            if torch.isinf(param.data).any():
                inf_parameters[name] = torch.isinf(param.data).nonzero(as_tuple=True)
        return inf_parameters

    # Example usage in a training loop
    def average_differences(self, differences):
        placeholder = Transformer(src_vocab_size=self.config['src_vocab_size'],
                                  tgt_vocab_size=self.config['tgt_vocab_size']
                                  , d_model=self.config['d_model'], num_heads=self.config['num_heads']
                                  , num_layers=self.config['num_layers'], d_ff=self.config['d_ff']
                                  , max_seq_length=self.config['max_seq_length'], dropout=self.config['dropout']
                                  , transformer_id=100)
        # placeholder.to(self.config['device'])
        placeholder.half()
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
            source_ids_validation = torch.tensor(self.dataset['source_ids_validation'],
                                                 device=self.config['device'])  # .to()
            target_ids_validation = torch.tensor(self.dataset['target_ids_validation'],
                                                 device=self.config['device'])  # .to(self.config['device'])
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
            del source_ids_validation, target_ids_validation
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
        output_t.half()
        # output_t.to(self.config['device'])
        for param in output_t.parameters():
            param.data.fill_(0)
        for server, param, output in zip(self.server_model.parameters(), loaded_transformer.parameters(),
                                         output_t.parameters()):
            # Check if the parameter is trainable.
            if server.requires_grad:
                # sum all transformer parameters
                output.data.copy_(server.data - param.data)
        return output_t
