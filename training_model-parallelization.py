import numpy as np

import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from fcn import FCN
from training_utils import R2_CHs,Random_Imaging,plot_debug

import os

import time
from datetime import datetime

import logging
import platform



if __name__ == "__main__":

    num_chemical_elements = 5


    training_folder_path = 'training_data_small/data/'
    test_folder_path = 'test_data_small/data/'

    training_results_folder_path = 'training_results/'
    debug_folder_path = training_results_folder_path +'debug/'
    weights_folder_path = training_results_folder_path +'weights/'
    train_learning_curve_folder_path = training_results_folder_path +'train_learning_curve/'

    test_results_folder_path = 'test_results/'
    test_learning_curve_folder_path = test_results_folder_path +'test_learning_curve/'

    if training_results_folder_path and not os.path.exists(training_results_folder_path):
        os.makedirs(training_results_folder_path)

    if debug_folder_path and not os.path.exists(debug_folder_path):
        os.makedirs(debug_folder_path)

    if weights_folder_path and not os.path.exists(weights_folder_path):
        os.makedirs(weights_folder_path)

    if train_learning_curve_folder_path and not os.path.exists(train_learning_curve_folder_path):
        os.makedirs(train_learning_curve_folder_path)

    if test_results_folder_path and not os.path.exists(test_results_folder_path):
        os.makedirs(test_results_folder_path)

    if test_learning_curve_folder_path and not os.path.exists(test_learning_curve_folder_path):
        os.makedirs(test_learning_curve_folder_path)

    num_training_data = len(os.listdir(training_folder_path))
    num_test_data = len(os.listdir(test_folder_path))

    training_data_path = training_folder_path + str('*.npy')
    test_data_path = test_folder_path + str('*.npy')

    train_dataset = tf.data.Dataset.list_files(training_data_path)
    test_dataset = tf.data.Dataset.list_files(test_data_path)

    batch_size = 2

    num_batches_train = num_training_data//batch_size

    train_dataset=train_dataset.shuffle(buffer_size = num_training_data).batch(batch_size = batch_size)
    test_dataset=test_dataset.batch(batch_size = batch_size)

    mp = False

    if mp:

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    hvd.init()

    config_proto = tf.compat.v1.ConfigProto()

    config_proto.gpu_options.visible_device_list = str(hvd.local_rank())

    config_proto.allow_soft_placement = True

    config_proto.log_device_placement = True

    config_proto.gpu_options.force_gpu_compatible = True

    config_proto.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

    sess = tf.compat.v1.InteractiveSession(config = config_proto)


    def get_model(FCN,input_shape, output_channels):

        input_channel = 1

        input_tensor = tf.keras.Input(shape = input_shape+(input_channel,))

        model = FCN(input_tensor, output_channels)

        return model

    input_shape = (256,256)

    output_channels = num_chemical_elements

    model = get_model(FCN,input_shape,output_channels)

    lr = 1e-3

    optimizer = tf.keras.optimizers.Adam(lr * hvd.size())

    if mp:

        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    loss_object = tf.keras.losses.MeanSquaredError()

    def get_batch(batch):

        batch = np.array(batch)

        batch_images=[]

        batch_labels=[]

        for i in range(len(batch)):

            data = np.load(batch[i])

            img = data[:,:,:,0]
            img = img.reshape(img.shape+(1,)).astype(np.float32)
            lbl = data[:,:,:,1:].astype(np.float32)

            rnd_imgng = Random_Imaging(image=img,labels=lbl)
            img,lbl = rnd_imgng.get_trasform()

            batch_images.append(img)

            batch_labels.append(lbl)

            plot_debug(img, lbl, i, debug_folder_path)

        batch_images = np.concatenate(batch_images)

        batch_labels = np.concatenate(batch_labels)

        return batch_images,batch_labels


    @tf.function
    def train_step(batch_images,batch_labels,first_batch):

        with tf.GradientTape() as tape:

            batch_predictions = model(batch_images, training=True)

            train_loss = loss_object(batch_labels, batch_predictions)

            if mp:

                scaled_train_loss = optimizer.get_scaled_loss(train_loss)

        tape = hvd.DistributedGradientTape(tape)

        if mp:

            scaled_gradients = tape.gradient(scaled_train_loss, model.trainable_variables)

            gradients = optimizer.get_unscaled_gradients(scaled_gradients)

        else:

            gradients = tape.gradient(train_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


        if first_batch:

            hvd.broadcast_variables(model.variables, root_rank=0)

            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        return train_loss


    @tf.function
    def test_step(batch_images,batch_labels,first_batch):

      batch_predictions = model(batch_images, training=False)

      test_loss = loss_object(batch_labels, batch_predictions)

      if first_batch:

            hvd.broadcast_variables(model.variables, root_rank=0)

            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

      return test_loss

    first_epoch = 424
    num_epochs = 500

    train_loss_learning_curve = []
    train_r2_learning_curve = []

    test_loss_learning_curve = []
    test_r2_learning_curve = []

    if first_epoch > 0:
        model.load_weights(weights_folder_path+'epoch-{}.h5'.format(first_epoch))

        train_loss_learning_curve = list(np.load(train_learning_curve_folder_path+'train_loss_learning_curve.npy'))
        train_r2_learning_curve = list(np.load(train_learning_curve_folder_path+'train_r2_learning_curve.npy'))
        test_loss_learning_curve = list(np.load(test_learning_curve_folder_path+'test_loss_learning_curve.npy'))
        test_r2_learning_curve = list(np.load(test_learning_curve_folder_path+'test_r2_learning_curve.npy'))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

    if hvd.rank() == 0:
        print("Running on host '{}'".format(platform.node()))
        print('Running on {} devices'.format(hvd.size()))

    for epoch in range(first_epoch,first_epoch + num_epochs):

        total_train_loss = 0.0

        total_average_r2_train = 0.0

        processed_batches_train = 0


        for train_batch_index,train_batch in enumerate(train_dataset.take(10000 // hvd.size())):

            batch_images,batch_labels = get_batch(train_batch)

            num_images = batch_images.shape[0]

            before = time.time()

            total_train_loss += train_step(batch_images,batch_labels,train_batch_index == 0)

            totaltime = time.time() - before

            processed_batches_train += 1

            train_loss = total_train_loss / processed_batches_train

            r2_CHs = R2_CHs(model,train_batch,num_chemical_elements)
            r2_all_elements_train = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_train += r2_all_elements_train[num_chemical_elements]
            r2_average_train = total_average_r2_train/processed_batches_train

            if (train_batch_index +1) % 1 == 0 and hvd.local_rank() == 0:

               print('Epoch [{}/{}] : Batch [{}/{}] : Train Loss = {:.4f}, Train R2 = {:.4f}, Processing Frequency = {:.1f} imgs/s'.format(epoch+1,
                                                                                first_epoch + num_epochs,
                                                                                train_batch_index +1,
                                                                                num_batches_train,
                                                                                train_loss,
                                                                                r2_average_train,
                                                                                num_images/totaltime))

        total_test_loss = 0.0

        total_average_r2_test = 0.0

        processed_batches_test = 0

        for test_batch_index,test_batch in enumerate(test_dataset.take(10000 // hvd.size())):

            processed_batches_test += 1

            batch_images,batch_labels = get_batch(test_batch)

            total_test_loss += test_step(batch_images,batch_labels,test_batch_index == 0)

            test_loss = total_test_loss/processed_batches_test

            r2_CHs = R2_CHs(model, test_batch, num_chemical_elements)
            r2_all_elements_test = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_test += r2_all_elements_test[num_chemical_elements]
            r2_average_test = total_average_r2_test/processed_batches_test

        print('Epoch [{}/{}]: Test Loss = {:.4f}, Test R2 = {:.4f}'.format(epoch+1,
                                                               first_epoch + num_epochs,
                                                               test_loss,
                                                               r2_average_test))


        train_loss_learning_curve.append(train_loss)
        train_loss_learning_curve_array = np.array(train_loss_learning_curve)

        train_r2_learning_curve.append(r2_all_elements_train)
        train_r2_learning_curve_array = np.array(train_r2_learning_curve)

        np.save(train_learning_curve_folder_path+'train_loss_learning_curve',train_loss_learning_curve_array)
        np.save(train_learning_curve_folder_path+'train_r2_learning_curve',train_r2_learning_curve_array)

        test_loss_learning_curve.append(test_loss)
        test_loss_learning_curve_array = np.array(test_loss_learning_curve)

        test_r2_learning_curve.append(r2_all_elements_test)
        test_r2_learning_curve_array = np.array(test_r2_learning_curve)

        np.save(test_learning_curve_folder_path+'test_loss_learning_curve',test_loss_learning_curve_array)
        np.save(test_learning_curve_folder_path+'test_r2_learning_curve',test_r2_learning_curve_array)


        #if hvd.rank() == 0:
        model.save_weights(weights_folder_path+'epoch-{}.h5'.format(epoch+1))
