import numpy as np

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from fcn import FCN
from training_utils import*

import os



if __name__ == "__main__":

    training_folder_path = '../training_data/data/'
    test_folder_path = '../test_data/data/'

    training_results_folder_path = 'results_data-parallelization/training_results/'
    make_results_folder(training_results_folder_path)

    test_results_folder_path = 'results_data-parallelization/test_results/'
    make_results_folder(test_results_folder_path, train=False)

    mp = False

    if mp:
        policy = mixed_precision.Policy('mixed_float16')

        mixed_precision.set_policy(policy)

    make_config()

    strategy = tf.distribute.MirroredStrategy()

    num_devices = strategy.num_replicas_in_sync

    batch_size_per_replica = 8

    global_batch_size = batch_size_per_replica * num_devices

    train_dataset, num_batches_train = make_dataset(training_folder_path, global_batch_size)
    test_dataset, _ = make_dataset(test_folder_path, global_batch_size)


    input_shape = (256,256)

    num_chemical_elements = 5

    output_channels = num_chemical_elements


    with strategy.scope():

        model = make_model(FCN, input_shape, output_channels)

        optimizer = tf.keras.optimizers.Adam()

        if mp:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

        loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


        def compute_loss(labels,predictions):

            per_example_loss = loss_object(labels, predictions)

            per_example_loss /= tf.cast(tf.reduce_prod(tf.shape(labels)[1:]), tf.float32)

            return tf.nn.compute_average_loss(per_example_loss, global_batch_size = labels.shape[0])


    first_epoch = 0
    num_epochs = 500

    save_every = 1

    train_loss_learning_curve = []
    train_r2_learning_curve = []

    test_loss_learning_curve = []
    test_r2_learning_curve = []

    if first_epoch > 0:

        model.load_weights(os.path.join(training_results_folder_path, 'weights/epoch-{}.h5'.format(first_epoch)))

        train_loss_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_loss_learning_curve.npy')))
        train_r2_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_r2_learning_curve.npy')))

        test_loss_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_loss_learning_curve.npy')))
        test_r2_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_r2_learning_curve.npy')))

    print_devices(num_devices)

    for epoch in range(first_epoch, first_epoch + num_epochs):

        total_train_loss = 0.0

        total_average_r2_train = 0.0

        processed_batches_train = 0

        for train_batch_index, train_batch in enumerate(train_dataset):

            train_images, train_labels = make_batch(train_batch)

            num_images = train_images.shape[0]

            train_loss, train_predictions = distributed_train_step(strategy,[train_images, train_labels], model, mp, loss_object, optimizer)

            total_train_loss = total_train_loss + train_loss

            processed_batches_train += 1

            train_loss = total_train_loss / processed_batches_train

            r2_CHs = R2_CHs(train_predictions, train_labels)
            r2_all_elements_train = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_train += r2_all_elements_train[num_chemical_elements]
            r2_average_train = total_average_r2_train / processed_batches_train

            plot_debug(train_images, train_labels, train_predictions,os.path.join(training_results_folder_path, 'debug/'))

            if (train_batch_index + 1) % 1 == 0:
                print('Epoch [{}/{}] : Batch [{}/{}] : Train Loss = {:.4f}, Train R2 = {:.4f}'.format(epoch + 1,
                                                                                                      first_epoch + num_epochs,
                                                                                                      train_batch_index + 1,
                                                                                                      num_batches_train,
                                                                                                      train_loss,
                                                                                                      r2_average_train))

        total_test_loss = 0

        total_average_r2_test = 0.0

        processed_batches_test = 0

        for test_batch_index, test_batch in enumerate(test_dataset):
            test_images, test_labels = make_batch(test_batch)

            test_loss, test_predictions = distributed_test_step(strategy,[test_images, test_labels], model, loss_object)

            plot_debug(test_images, test_labels, test_predictions,os.path.join(test_results_folder_path, 'debug/'))

            total_test_loss = total_test_loss + test_loss

            processed_batches_test += 1

            test_loss = total_test_loss / processed_batches_test

            r2_CHs = R2_CHs(test_predictions, test_labels)
            r2_all_elements_test = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_test += r2_all_elements_test[num_chemical_elements]
            r2_average_test = total_average_r2_test / processed_batches_test

        print('Epoch [{}/{}]: Test Loss = {:.4f}, Test R2 = {:.4f}'.format(epoch + 1,
                                                                           first_epoch + num_epochs,
                                                                           test_loss,
                                                                           r2_average_test))

        train_loss_learning_curve.append(train_loss)
        train_r2_learning_curve.append(r2_all_elements_train)

        np.save(os.path.join(training_results_folder_path, 'learning_curve/train_loss_learning_curve'),np.array(train_loss_learning_curve))
        np.save(os.path.join(training_results_folder_path, 'learning_curve/train_r2_learning_curve'),np.array(train_r2_learning_curve))

        test_loss_learning_curve.append(test_loss)
        test_r2_learning_curve.append(r2_all_elements_test)

        np.save(os.path.join(test_results_folder_path, 'learning_curve/test_loss_learning_curve'),np.array(test_loss_learning_curve))
        np.save(os.path.join(test_results_folder_path, 'learning_curve/test_r2_learning_curve'),np.array(test_r2_learning_curve))

        plot_learning_curves(np.array(train_loss_learning_curve),
                             np.array(test_loss_learning_curve),
                             np.array(train_r2_learning_curve),
                             np.array(test_r2_learning_curve),
                             path='results_data-parallelization/')

        if epoch % save_every == 0:
            model.save_weights(os.path.join(training_results_folder_path, 'weights/epoch-{}.h5'.format(epoch + 1)))
