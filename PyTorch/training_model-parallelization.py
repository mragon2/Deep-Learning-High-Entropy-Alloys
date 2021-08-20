import numpy as np
import numpy as np

import torch
import horovod.torch as hvd

from training_utils import*

import os




if __name__ == "__main__":

    hvd.init()

    training_folder_path = '../training_data/data/'
    test_folder_path = '../test_data/data/'

    training_results_folder_path = 'results_model-parallelization/training_results/'
    test_results_folder_path = 'results_model-parallelization/test_results/'

    if hvd.local_rank() == 0:

        make_results_folder(training_results_folder_path)
        make_results_folder(test_results_folder_path, train=False)


    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())

    batch_size = 8

    train_dataset, num_batches_train = make_dataset_hvd(training_folder_path, batch_size, hvd)
    test_dataset, _ = make_dataset_hvd(test_folder_path, batch_size, hvd)

    num_chemical_elements = 5

    output_channels = num_chemical_elements

    model = make_model(output_channels)

    criterion = torch.nn.MSELoss()

    mp = False

    if mp:

        loss_scaler = torch.cuda.amp.GradScaler()


    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.01, amsgrad = True)

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters = model.named_parameters())


    hvd.broadcast_parameters(model.state_dict(), root_rank = 0)


    if hvd.local_rank() == 0:

        print_devices()


    first_epoch = 0

    num_epochs = 500

    save_every = 1

    train_loss_learning_curve = []
    train_r2_learning_curve = []

    test_loss_learning_curve = []
    test_r2_learning_curve = []

    if first_epoch > 0:

        model.load_state_dict(torch.load(os.path.join(training_results_folder_path, 'weights/epoch-{}.pkl'.format(first_epoch))))

        train_loss_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_loss_learning_curve.npy'),allow_pickle=True))
        train_r2_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_r2_learning_curve.npy'),allow_pickle=True))

        test_loss_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_loss_learning_curve.npy'),allow_pickle=True))
        test_r2_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_r2_learning_curve.npy'),allow_pickle=True))

    for epoch in range(first_epoch,first_epoch + num_epochs):

        total_train_loss = 0.0

        total_average_r2_train = 0.0

        processed_batches_train = 0

        for train_batch_index,train_batch in enumerate(train_dataset):

            train_batch_images, train_batch_labels = make_batch(train_batch)

            train_loss, train_batch_predictions = train_step([train_batch_images, train_batch_labels], model, mp,criterion, optimizer)

            if torch.cuda.is_available():

                train_batch_images, train_batch_labels = train_batch_images.cpu(), train_batch_labels.cpu()

            plot_debug(train_batch_images, train_batch_labels, train_batch_predictions,os.path.join(training_results_folder_path, 'debug/'))

            total_train_loss += train_loss

            processed_batches_train += 1

            train_loss = total_train_loss / processed_batches_train

            r2_CHs = R2_CHs(train_batch_predictions, train_batch_labels)
            r2_all_elements_train = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_train += r2_all_elements_train[num_chemical_elements]
            r2_average_train = total_average_r2_train / processed_batches_train

            if (train_batch_index +1) % 1 and hvd.local_rank() == 0:

                print('Epoch [{}/{}] : Batch [{}/{}] : Train Loss = {:.4f}, Train R2 = {:.4f}'.format(epoch + 1,
                                                                                                      first_epoch + num_epochs,
                                                                                                      train_batch_index + 1,
                                                                                                      num_batches_train,
                                                                                                      train_loss,
                                                                                                      r2_average_train))


        total_test_loss = 0.0

        total_average_r2_test = 0.0

        processed_batches_test = 0


        for test_batch_index,test_batch in enumerate(test_dataset):

            test_batch_images, test_batch_labels = make_batch(test_batch)

            test_loss, test_batch_predictions = test_step([test_batch_images, test_batch_labels], model, mp, criterion)

            if torch.cuda.is_available():

                test_batch_images, test_batch_labels = test_batch_images.cpu(), test_batch_labels.cpu()

            plot_debug(test_batch_images, test_batch_labels, test_batch_predictions,os.path.join(test_results_folder_path, 'debug/'))

            total_test_loss += test_loss

            processed_batches_test += 1

            test_loss = total_test_loss / processed_batches_test

            r2_CHs = R2_CHs(test_batch_predictions, test_batch_labels)
            r2_all_elements_test = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_test += r2_all_elements_test[num_chemical_elements]
            r2_average_test = total_average_r2_test / processed_batches_test

        if hvd.local_rank() == 0:

            print('Epoch [{}/{}]: Test Loss = {:.4f}, Test R2 = {:.4f}'.format(epoch+1,
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
                             path='results_model-parallelization/')

        if epoch % save_every == 0:
            torch.save(model.state_dict(),os.path.join(training_results_folder_path, 'weights/epoch-{}.pkl'.format(epoch + 1)))
