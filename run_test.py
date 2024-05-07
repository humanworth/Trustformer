# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:48:15 2024

@author: aliab
"""

import Utils
import Federated_training

config = Utils.load_config("config.json")

# base_seal_folder = "sealed_models"
paths = ["sealed_models/worker0", "sealed_models/worker1", "sealed_models/worker2"]
####################### Initialize workers based on the number specified in config dictionary
workers = Federated_training.initialize_workers(config)

main_server = Federated_training.initialize_server(config)


# for epoch in range(config['n_epochs']):
#     print("start training for epoch {}".format(epoch))
#     Federated_training.train_workers(workers)
#     paths = Federated_training.seal_store_models(workers)
#     server = main_server.aggregate_models(paths, agg_method='Avg')  # FL aggregation happens here
#     new_optimizer = main_server.aggregate_optimizers([worker.optimizer for worker in workers])
#     Federated_training.setup_optimizers(new_optimizer, optimizer_name='adam', workers=workers)
#     Federated_training.send_global_model_to_clients(config,server=server['model']) #send aggregated model to clients
#     workers = Federated_training.load_unseal_models(paths, workers) # this function is equal to receieve model in client side
#     Federated_training.store_worker_info(workers, epoch)


workers = Federated_training.load_unseal_models(paths, workers) # this function is equal to receieve model in client side

print(f'Starting Global Model Evaluation with {config["test_in_each_worker"]} number of data fo each worker')

results = Federated_training.evaluate_workers(workers, store=True)

