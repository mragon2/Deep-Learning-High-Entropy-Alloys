#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import torch
import horovod.torch as hvd
from fcn import FCN
from training_utils import*


if __name__ == "__main__":

    # to run:
    # horovodrun -np 4 -H localhost:4 python training_model_parallelization.py

    results_folder_path = 'results_model-parallelization/'
    data_folder_path = '../../HEA_simulations/data/'

    hvd.init()
    num_devices = hvd.size()

    if hvd.local_rank() == 0:
        print('Running model with horovod technique')

    batch_size = 8

    input_shape = (256, 256)
    num_chemical_elements = 5
    mp = False

    model = FCN(output_channels = num_chemical_elements)

    if torch.cuda.is_available():

        torch.cuda.set_device(hvd.local_rank())
        model.to('cuda')

    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.01, amsgrad = True)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters = model.named_parameters())
    criterion = torch.nn.MSELoss()

    hvd.broadcast_parameters(model.state_dict(), root_rank = 0)

    first_epoch = 0
    num_epochs = 500
    save_every = 1

    train_model_parallel(results_folder_path,
                         data_folder_path,
                         hvd,
                         num_devices,
                         batch_size,
                         model,
                         mp,
                         optimizer,
                         criterion,
                         num_chemical_elements,
                         first_epoch,
                         num_epochs,
                         save_every)















