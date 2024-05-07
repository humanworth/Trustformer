# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:48:15 2024

@author: aliab
"""


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
    server = main_server.aggregate_models(paths, agg_method='Avg') # FL aggregation happens here
    new_optimizer = main_server.aggregate_optimizers([worker.optimizer for worker in workers])
    Federated_training.setup_optimizers(new_optimizer, optimizer_name='adam',workers=workers)
    # Federated_training.setup_optimizers(new_optimizer, workers=workers, optimizer_name = 'adam')

    # Federated_training.send_global_model_to_clients(config,server=server['model']) #send aggregated model to clients
    # workers = Federated_training.load_unseal_models(paths, workers) # this function is equal to receieve model in client side
    # Federated_training.store_worker_info(workers, epoch)

#load_workers[0].start_training()



















######################## Juts for test


    
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
