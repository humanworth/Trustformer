# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:45:23 2024

@author: aliab
"""
from Transformer import Transformer
import Dataset as ds
from Worker import Worker
import numpy as np
import Utils
import os
import json
config = Utils.load_config("config.json")
from Server import Server
def train_workers(workers):
    print("start training workers...")
    for worker in workers:
        worker.start_training()
        

def evaluate_workers(workers):
    print("start evaluating workers...")
    for worker in workers:
        worker.evaluate_model()

def seal_store_models(workers):
    path = []
    print("Sealing and storing workers models...")
    for worker in workers:
        path.append(worker.encrypt_store_model())
    return path
    

def load_unseal_models(paths, workers):
    new_workers = []
    print("receiving info for workers...")
    for path, worker in zip(paths, workers):
        new_workers.append(worker.load_decrypt_model(path))
    
    return new_workers
def setup_optimizers(workers, optimizer_name='sgd'):
    for worker in workers:
        worker.set_optimizer(optimizer_name=optimizer_name)

def initialize_server(config):
    return Server(config = config)
    
    
    
def adjust_parameters(transformer, operator = "bigger"):
    parameters=[]
    # Print the names and shapes of the model's parameters
    for name, param in transformer.named_parameters():
        param.requires_grad = False
        #print(f"Name: {name}")
        #print(f"Params: {param}")
        parameters.append(param.data.cpu().detach().numpy())
        
        len(parameters)
    stds=[]
    for params in parameters:
        stds.append(np.average(np.std(params,axis=0)))
    avgStd=np.average(stds)
    if operator == "bigger":
        paramsToFineTune = stds>avgStd
    elif operator == "lower":
        paramsToFineTune = stds<avgStd
    named_parameters_iter = list(transformer.named_parameters())
    selected_parameters = []
    for select, (name, param) in zip(paramsToFineTune, named_parameters_iter):
        if select:
            param.requires_grad = True
            
def initialize_workers(config):
    workers=[]

    for i in range(config['n_workers']):
        print("Initializing worker ", i)
        trans =  Transformer(src_vocab_size = config['src_vocab_size'], tgt_vocab_size= config['tgt_vocab_size']
                             , d_model = config['d_model'], num_heads = config['num_heads']
                             , num_layers = config['num_layers'], d_ff = config['d_ff']
                             , max_seq_length = config['max_seq_length'], dropout = config['dropout']
                             ,transformer_id=i)
        data = ds.tokenized_dataset(name='wmt19', n_records_train = (i+1) * config['data_in_each_worker'], n_records_test = (i+1) * config['test_in_each_worker'], max_seq_length = config['max_seq_length'], train_offset = i * config['data_in_each_worker'], test_offset = i * config['test_in_each_worker']) 
        workers.append(Worker(data,config, 
                              trans, name = 'worker'+str(i),
                              provisioning_key = b'Sixteen byte key'))

    return workers

def send_global_model_to_clients(config, server):
    n_clients = config['n_workers']
    for i in range(n_clients):
        name = "worker{}".format(i)
        server.encrypt_store_model(name=name)


def store_worker_info(workers, epoch):
    results_folder = config['results']
    for worker in workers:
        results_folder= config['results']+f'/{epoch+1}/{worker.name}'#.format(worker.name)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Define the file path to save the data
        file_path = os.path.join(results_folder, f'{worker.name}.json')

        # Write the dictionary data to a JSON file
        with open(file_path, 'w') as file:
            json.dump(worker.history, file)
        print(f"history of worker {worker.name} epoch {epoch+1} saved to: {file_path}")
