#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import tensorflow as tf
from fcn import FCN
from training_utils import*


if __name__ == "__main__":

    results_folder_path = 'results_data-parallelization/'
    data_folder_path = '../data/'

    strategy = tf.distribute.MirroredStrategy(['/cpu:0','/cpu:1','/cpu:2'])

    num_devices = strategy.num_replicas_in_sync

    print('Running model with mirrored strategy strategy technique')

    batch_size_per_replica = 2
    global_batch_size = batch_size_per_replica * num_devices

    input_shape = (256,256)
    num_chemical_elements = 5
    mp = False

    with strategy.scope():

        model = make_model(FCN, input_shape, output_channels = num_chemical_elements)
        lr = 1e-3
        optimizer = tf.keras.optimizers.Adam(lr = lr)
        loss_object = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.NONE)

    first_epoch = 0
    num_epochs = 500
    save_every = 1

    train_data_parallel(results_folder_path,
                        data_folder_path,
                        strategy,
                        num_devices,
                        global_batch_size,
                        model,
                        mp,
                        optimizer,
                        loss_object,
                        num_chemical_elements,
                        first_epoch,
                        num_epochs,
                        save_every)

