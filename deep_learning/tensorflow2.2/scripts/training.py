#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import tensorflow as tf
from fcn import FCN
from training_utils import*


if __name__ == "__main__":

    results_folder_path = 'results/'
    data_folder_path = '../../HEA_simulations/data/'

    num_devices = 1

    print_devices(num_devices)
    make_config()

    batch_size = 8

    input_shape = (256,256)
    num_chemical_elements = 5
    mp = False

    model = make_model(FCN, input_shape, output_channels = num_chemical_elements)
    lr = 1e-3
    optimizer = tf.keras.optimizers.Adam(lr = lr, amsgrad = True)
    loss_object = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.NONE)

    first_epoch = 0
    num_epochs = 500
    save_every = 1

    train(results_folder_path,
          data_folder_path,
          batch_size,
          model,
          mp,
          optimizer,
          loss_object,
          num_chemical_elements,
          first_epoch,
          num_epochs,
          save_every)

