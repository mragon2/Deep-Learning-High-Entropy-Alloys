#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import torch
from fcn import FCN
from training_utils import*


if __name__ == "__main__":

    results_folder_path = 'results/'
    data_folder_path = '../../HEA_simulations/data/'

    num_devices = 1

    batch_size = 8

    input_shape = (256, 256)
    num_chemical_elements = 5
    mp = False

    model = FCN(output_channels = num_chemical_elements)
    if torch.cuda.is_available():
        model.to('cuda')
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.01, amsgrad = True)
    criterion = torch.nn.MSELoss()

    first_epoch = 0
    num_epochs = 500
    save_every = 1

    train(results_folder_path,
          data_folder_path,
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
