import numpy as np
import torch
import torchvision
import torch.nn as nn

from fcn import FCN

from training_utils import R2_CHs,Random_Imaging,plot_debug

import os
from glob import glob

import time
from datetime import datetime

import logging
import platform



if __name__ == "__main__":

    training_folder_path = '../training_data-try/data/'
    test_folder_path = '../test_data-try/data/'

    training_results_folder_path = 'results_data-parallelization/training_results/'
    training_debug_folder_path = training_results_folder_path +'debug/'
    weights_folder_path = training_results_folder_path +'weights/'
    train_learning_curve_folder_path = training_results_folder_path +'train_learning_curve/'

    test_results_folder_path = 'results_data-parallelization/test_results/'
    test_learning_curve_folder_path = test_results_folder_path +'test_learning_curve/'


    if training_results_folder_path and not os.path.exists(training_results_folder_path):
        os.makedirs(training_results_folder_path)

    if training_debug_folder_path and not os.path.exists(training_debug_folder_path):
        os.makedirs(training_debug_folder_path)

    if weights_folder_path and not os.path.exists(weights_folder_path):
        os.makedirs(weights_folder_path)

    if train_learning_curve_folder_path and not os.path.exists(train_learning_curve_folder_path):
        os.makedirs(train_learning_curve_folder_path)

    if test_results_folder_path and not os.path.exists(test_results_folder_path):
        os.makedirs(test_results_folder_path)

    if test_learning_curve_folder_path and not os.path.exists(test_learning_curve_folder_path):
        os.makedirs(test_learning_curve_folder_path)


    mp = False

    train_dataset = np.sort(np.array(glob(training_folder_path +'data*.npy')))
    test_dataset = np.sort(np.array(glob(test_folder_path +'data*.npy')))

    num_training_data = len(os.listdir(training_folder_path))
    num_test_data = len(os.listdir(test_folder_path))

    num_chemical_elements = 5

    batch_size = 4

    train_loader=torch.utils.data.DataLoader(train_dataset,
                                            shuffle = True,
                                            batch_size = batch_size)

    num_batches_train= len(train_dataset)//batch_size

    test_loader=torch.utils.data.DataLoader(test_dataset,
                                            batch_size = 1)

    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = FCN()

    if torch.cuda.is_available():

        model.to(devices)

        model = nn.DataParallel(model)

        num_devices = torch.cuda.device_count()

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.01, amsgrad = True)

    if mp:
        losss_scaler = torch.cuda.amp.GradScaler()


    def get_batch(batch):

        batch = np.array(batch)

        batch_images = []
        batch_labels = []

        for i in range(len(batch)):

            data = np.load(batch[i])

            img = data[0,:,:,0]
            img = img.reshape(1,1,img.shape[0],img.shape[1])

            lbl = data[0,:,:,1:]
            lbl_torch = []

            for l in range(lbl.shape[2]):

                lbl_element = lbl[:,:,l]

                lbl_torch.append([lbl_element])

            lbl_torch = np.array(lbl_torch)

            lbl_torch = lbl_torch.reshape(1,lbl.shape[2],lbl.shape[0],lbl.shape[1])

            rnd_imgng = Random_Imaging(image=img,labels=lbl_torch)
            img,lbl_torch = rnd_imgng.get_trasform()

            plot_debug(img,lbl_torch,i,training_debug_folder_path)

            batch_images.append(img)
            batch_labels.append(lbl_torch)

        batch_images = torch.Tensor(np.concatenate(batch_images))

        batch_labels = torch.Tensor(np.concatenate(batch_labels))

        if torch.cuda.is_available():

            batch_images = batch_images.cuda()

            batch_labels = batch_labels.cuda()


        return batch_images,batch_labels


    def train_step(batch):

        batch_images,batch_labels  = batch

        model.zero_grad()

        if mp:

            with torch.cuda.amp.autocast():

                batch_predictions = model(batch_images)

                train_loss = criterion(batch_predictions,batch_labels)

                loss_scaler.scale(train_loss).backward()

                loss_scaler.step(optimizer)

                loss_scaler.update()

        else:

            batch_predictions = model(batch_images)

            train_loss = criterion(batch_predictions,batch_labels)

            train_loss.backward()

            optimizer.step()

        batch_images = batch_images.detach().cpu().numpy()

        batch_labels = batch_labels.detach().cpu().numpy()

        batch_predictions = batch_predictions.detach().cpu().numpy()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        return train_loss,batch_predictions

    def test_step(batch):

        batch_images,batch_labels  = batch

        with torch.no_grad():

            if mp:

                with torch.cuda.amp.autocast():

                    batch_predictions = model(batch_images)

            else:

                batch_predictions = model(batch_images)

        test_loss = criterion(batch_predictions,batch_labels)


        batch_images = batch_images.detach().cpu().numpy()

        batch_labels = batch_labels.detach().cpu().numpy()

        batch_predictions = batch_predictions.detach().cpu().numpy()


        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        return test_loss,batch_predictions


    print('Running on host {}'.format(platform.node()))
    if torch.cuda.is_available():

        print('Running on {} devices'.format(num_devices))

    else:

        print('Running on 1 cpu')

    first_epoch = 0

    num_epochs = 500

    train_loss_learning_curve = []
    train_r2_learning_curve = []

    test_loss_learning_curve = []
    test_r2_learning_curve = []

    if first_epoch > 0:

        model.load_state_dict(torch.load(weights_folder_path+'epoch-{}.pkl'.format(first_epoch)))


        train_loss_learning_curve = list(np.load(train_learning_curve_folder_path+'train_loss_learning_curve.npy',allow_pickle=True))
        train_r2_learning_curve = list(np.load(train_learning_curve_folder_path+'train_r2_learning_curve.npy',allow_pickle=True))
        test_loss_learning_curve = list(np.load(test_learning_curve_folder_path+'test_loss_learning_curve.npy',allow_pickle=True))
        test_r2_learning_curve = list(np.load(test_learning_curve_folder_path+'test_r2_learning_curve.npy',allow_pickle=True))


    for epoch in range(first_epoch,first_epoch + num_epochs):

        total_train_loss = 0.0

        total_average_r2_train = 0.0

        processed_batches_train = 0

        for train_batch_index,train_batch in enumerate(train_loader):

            train_batch_images, train_batch_labels = get_batch(train_batch)

            num_images =  train_batch_images.shape[0]

            before = time.time()

            train_loss,train_batch_predictions = train_step([train_batch_images, train_batch_labels])

            total_train_loss += train_loss

            processed_batches_train += 1

            train_loss = total_train_loss / processed_batches_train

            r2_CHs = R2_CHs(train_batch_predictions, train_batch_labels)
            r2_all_elements_train = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_train += r2_all_elements_train[num_chemical_elements]
            r2_average_train = total_average_r2_train/processed_batches_train

            totaltime = time.time() - before

            if (train_batch_index +1) % 1 == 0:


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


        for test_batch_index,test_batch in enumerate(test_loader):

            test_batch_images, test_batch_labels = get_batch(test_batch)

            test_loss, test_batch_predictions = test_step([test_batch_images, test_batch_labels])

            total_test_loss += test_loss

            processed_batches_test += 1

            test_loss = total_test_loss / processed_batches_test

            r2_CHs = R2_CHs(test_batch_predictions, test_batch_labels)
            r2_all_elements_test = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_test += r2_all_elements_test[num_chemical_elements]
            r2_average_test = total_average_r2_test / processed_batches_test


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

        torch.save(model.state_dict(),weights_folder_path+'epoch-{}.pkl'.format(epoch+1))
