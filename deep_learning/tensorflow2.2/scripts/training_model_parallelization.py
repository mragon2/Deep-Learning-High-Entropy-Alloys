#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import tensorflow as tf
import horovod.tensorflow as hvd
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
        print_devices(num_devices)

    make_config_hvd(hvd)

    batch_size = 8

    input_shape = (256, 256)
    num_chemical_elements = 5
    mp = True

    model = make_model(FCN, input_shape, output_channels=num_chemical_elements)
    lr = 1e-3
    optimizer = tf.keras.optimizers.Adam(lr = lr * num_devices)
    loss_object = tf.keras.losses.MeanSquaredError()

    first_epoch = 0
    num_epochs = 500
    save_every = 1

    train_model_parallel(results_folder_path,
                         data_folder_path,
                         hvd,
                         batch_size,
                         model,
                         mp,
                         optimizer,
                         loss_object,
                         num_chemical_elements,
                         first_epoch,
                         num_epochs,
                         save_every)









