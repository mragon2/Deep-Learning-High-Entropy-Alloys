#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import numpy as np

import torch
import torchvision
import torch.nn as nn

from fcn import FCN

import os
from glob import glob

from skimage.feature import peak_local_max
from skimage.filters import gaussian

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import platform


def print_devices(num_devices):

    print('Running on host {}'.format(platform.node()))

    if torch.cuda.is_available():

        if num_devices == 1:

            print('Running on {} GPU'.format(num_devices))

        else:

            print('Running on {} GPUs'.format(num_devices))

    else:

        if num_devices == 1:

            print('Running on {} CPU'.format(num_devices))

        else:

            print('Running on {} CPUs'.format(num_devices))



def make_folder(path):
    if path and not os.path.exists(path):
        os.makedirs(path)

def make_results_folder(path,train = True):

    make_folder(path)

    debug_path = os.path.join(path, 'debug/')
    make_folder(debug_path)

    if train:

        weights_path = os.path.join(path, 'weights/')
        make_folder(weights_path)

    learning_curve_path = os.path.join(path, 'learning_curve/')
    make_folder(learning_curve_path)

def make_dataset(path,batch_size):

    dataset = np.sort(np.array(glob(os.path.join(path,'img_lbl/HEA_img_lbl*.npy'))))

    num_batches = len(dataset) // batch_size

    dataset = torch.utils.data.DataLoader(dataset,shuffle = True, batch_size = batch_size)

    return dataset,num_batches


def make_dataset_hvd(path, batch_size, hvd):

    dataset = np.sort(np.array(glob(os.path.join(path, 'img_lbl/HEA_img_lbl*.npy'))))

    num_batches = len(dataset) // batch_size

    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                   num_replicas = hvd.size(),
                                                                   rank = hvd.rank())

    dataset = torch.utils.data.DataLoader(dataset,
                                          shuffle = False,
                                          batch_size = batch_size,
                                          sampler = data_sampler)


    return dataset,num_batches

def make_batch(batch):

    batch = np.array(batch)

    batch_images = []
    batch_labels = []

    for i in range(len(batch)):

        data = np.load(batch[i])

        img = data[0,:,:,0]
        img = img.reshape(1,1,img.shape[0],img.shape[1])

        lbl = data[0, :, :, 1:]
        lbl_torch = []

        for l in range(lbl.shape[2]):

            lbl_element = lbl[:, :, l]

            lbl_torch.append([lbl_element])

        lbl_torch = np.array(lbl_torch)

        lbl_torch = lbl_torch.reshape(1, lbl.shape[2], lbl.shape[0], lbl.shape[1])

        rnd_imgng = Random_Imaging(image=img,labels=lbl_torch)
        img,lbl_torch = rnd_imgng.get_trasform()

        batch_images.append(img)
        batch_labels.append(lbl_torch)

    batch_images = torch.Tensor(np.concatenate(batch_images))

    batch_labels = torch.Tensor(np.concatenate(batch_labels))

    if torch.cuda.is_available():

        batch_images = batch_images.cuda()

        batch_labels = batch_labels.cuda()


    return batch_images,batch_labels


def detatch(x,mp):

    x = x.detach().cpu().numpy()

    if mp:

        x = x.astype('float32')

    return x


def train_step(batch,model,mp,criterion,optimizer):

    batch_images,batch_labels  = batch

    model.zero_grad()

    if mp:

        loss_scaler = torch.cuda.amp.GradScaler()

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

    batch_predictions = detatch(batch_predictions,mp)

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return train_loss,batch_predictions

def test_step(batch,model,mp,criterion):

    batch_images,batch_labels  = batch

    with torch.no_grad():

        if mp:

            with torch.cuda.amp.autocast():

                batch_predictions = model(batch_images)

        else:

            batch_predictions = model(batch_images)

    test_loss = criterion(batch_predictions,batch_labels)

    batch_predictions = detatch(batch_predictions, mp)

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return test_loss,batch_predictions


def train(results_folder_path,
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
          save_every):

    training_results_folder_path = os.path.join(results_folder_path, 'training_results/')
    if hvd.local_rank() == 0:
        make_results_folder(training_results_folder_path)

    test_results_folder_path = os.path.join(results_folder_path, 'test_results/')
    if hvd.local_rank() == 0:
        make_results_folder(test_results_folder_path, train=False)

    training_data_folder_path = os.path.join(data_folder_path, 'training_data/')
    test_data_folder_path = os.path.join(data_folder_path, 'test_data/')

    train_dataset, num_batches_train = make_dataset(training_data_folder_path, batch_size)
    test_dataset, _ = make_dataset(test_data_folder_path, batch_size)

    print_devices(num_devices)

    if mp:
        print('Running model with mixed precision technique')

    train_loss_learning_curve = []
    train_r2_learning_curve = []

    test_loss_learning_curve = []
    test_r2_learning_curve = []


    if first_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(training_results_folder_path, 'weights/epoch-{}.pkl'.format(first_epoch))))

        train_loss_learning_curve = list(np.load(os.path.join(training_results_folder_path,'learning_curve/train_loss_learning_curve.npy'),allow_pickle=True))
        train_r2_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_r2_learning_curve.npy'),allow_pickle=True))

        test_loss_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_loss_learning_curve.npy'),allow_pickle=True))
        test_r2_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_r2_learning_curve.npy'),allow_pickle=True))

    for epoch in range(first_epoch, first_epoch + num_epochs):

        total_train_loss = 0.0

        total_average_r2_train = 0.0

        processed_batches_train = 0

        for train_batch_index, train_batch in enumerate(train_dataset):

            train_batch_images, train_batch_labels = make_batch(train_batch)

            train_loss, train_batch_predictions = train_step([train_batch_images, train_batch_labels], model, mp,
                                                             criterion, optimizer)

            if torch.cuda.is_available():
                train_batch_images, train_batch_labels = train_batch_images.cpu(), train_batch_labels.cpu()

            plot_debug(train_batch_images, train_batch_labels, train_batch_predictions,
                       os.path.join(training_results_folder_path, 'debug/'))

            total_train_loss += train_loss

            processed_batches_train += 1

            train_loss = total_train_loss / processed_batches_train

            r2_CHs = R2_CHs(train_batch_predictions, train_batch_labels,num_chemical_elements)
            r2_all_elements_train = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_train += r2_all_elements_train[num_chemical_elements]
            r2_average_train = total_average_r2_train / processed_batches_train

            if (train_batch_index + 1) % 1 == 0:

                print('Epoch [{}/{}] : Batch [{}/{}] : Train Loss = {:.4f}, Train R2 = {:.4f}'.format(epoch + 1,
                                                                                                      first_epoch + num_epochs,
                                                                                                      train_batch_index + 1,
                                                                                                      num_batches_train + 1,
                                                                                                      train_loss,
                                                                                                      r2_average_train))

        total_test_loss = 0.0

        total_average_r2_test = 0.0

        processed_batches_test = 0

        for test_batch_index, test_batch in enumerate(test_dataset):

            test_batch_images, test_batch_labels = make_batch(test_batch)

            test_loss, test_batch_predictions = test_step([test_batch_images, test_batch_labels], model, mp, criterion)

            if torch.cuda.is_available():
                test_batch_images, test_batch_labels = test_batch_images.cpu(), test_batch_labels.cpu()

            plot_debug(test_batch_images, test_batch_labels, test_batch_predictions,
                       os.path.join(test_results_folder_path, 'debug/'))

            total_test_loss += test_loss

            processed_batches_test += 1

            test_loss = total_test_loss / processed_batches_test

            r2_CHs = R2_CHs(test_batch_predictions, test_batch_labels,num_chemical_elements)
            r2_all_elements_test = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_test += r2_all_elements_test[num_chemical_elements]
            r2_average_test = total_average_r2_test / processed_batches_test

        print('Epoch [{}/{}]: Test Loss = {:.4f}, Test R2 = {:.4f}'.format(epoch + 1,
                                                                           first_epoch + num_epochs,
                                                                           test_loss,
                                                                           r2_average_test))

        save_epoch_results(training_results_folder_path,
                           test_results_folder_path,
                           epoch,
                           save_every,
                           model,
                           train_loss_learning_curve,
                           train_r2_learning_curve,
                           test_loss_learning_curve,
                           test_r2_learning_curve,
                           train_loss,
                           r2_all_elements_train,
                           test_loss,
                           r2_all_elements_test)



def train_model_parallel(results_folder_path,
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
                         save_every):

    training_results_folder_path = os.path.join(results_folder_path, 'training_results/')
    test_results_folder_path = os.path.join(results_folder_path, 'test_results/')


    training_data_folder_path = os.path.join(data_folder_path, 'training_data/')
    test_data_folder_path = os.path.join(data_folder_path, 'test_data/')

    train_dataset, num_batches_train = make_dataset_hvd(training_data_folder_path, batch_size, hvd)
    test_dataset, _ = make_dataset_hvd(test_data_folder_path, batch_size, hvd)

    if hvd.local_rank() == 0:

        make_results_folder(training_results_folder_path)
        make_results_folder(test_results_folder_path, train = False)

        print_devices(num_devices)

    if mp:

        print('Running model with mixed precision technique')

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

    for epoch in range(first_epoch, first_epoch + num_epochs):

        total_train_loss = 0.0

        total_average_r2_train = 0.0

        processed_batches_train = 0

        for train_batch_index, train_batch in enumerate(train_dataset):

            train_batch_images, train_batch_labels = make_batch(train_batch)

            train_loss, train_batch_predictions = train_step([train_batch_images, train_batch_labels], model, mp,
                                                             criterion, optimizer)

            if torch.cuda.is_available():
                train_batch_images, train_batch_labels = train_batch_images.cpu(), train_batch_labels.cpu()

            plot_debug(train_batch_images, train_batch_labels, train_batch_predictions,
                       os.path.join(training_results_folder_path, 'debug/'))

            total_train_loss += train_loss

            processed_batches_train += 1

            train_loss = total_train_loss / processed_batches_train

            r2_CHs = R2_CHs(train_batch_predictions, train_batch_labels,num_chemical_elements)
            r2_all_elements_train = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_train += r2_all_elements_train[num_chemical_elements]
            r2_average_train = total_average_r2_train / processed_batches_train

            if (train_batch_index + 1) % 1 == 0 and hvd.local_rank() == 0:

                print('Epoch [{}/{}] : Batch [{}/{}] : Train Loss = {:.4f}, Train R2 = {:.4f}'.format(epoch + 1,
                                                                                                      first_epoch + num_epochs,
                                                                                                      train_batch_index + 1,
                                                                                                      num_batches_train + 1,
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

            r2_CHs = R2_CHs(test_batch_predictions, test_batch_labels,num_chemical_elements)
            r2_all_elements_test = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_test += r2_all_elements_test[num_chemical_elements]
            r2_average_test = total_average_r2_test / processed_batches_test

        if hvd.local_rank() == 0:

            print('Epoch [{}/{}]: Test Loss = {:.4f}, Test R2 = {:.4f}'.format(epoch+1,
                                                                                first_epoch + num_epochs,
                                                                                test_loss,
                                                                                r2_average_test))

        save_epoch_results(training_results_folder_path,
                           test_results_folder_path,
                           epoch,
                           save_every,
                           model,
                           train_loss_learning_curve,
                           train_r2_learning_curve,
                           test_loss_learning_curve,
                           test_r2_learning_curve,
                           train_loss,
                           r2_all_elements_train,
                           test_loss,
                           r2_all_elements_test)




class Random_Imaging():

    """
    * Random_Imaging: class to perform random transformations on the images.
    The random transformations include random blur, random brightness, random contrast,
    random gamma and random flipping/rotation. Input parameters:

    - image: image to which apply the random transformations.
    - labels: included for the random flipping/rotation.

    """

    def __init__ (self,image,labels):

        self.image = image
        self.labels = labels

    def random_blur(self,image,low,high):

        sigma = np.random.uniform(low,high)

        image = gaussian(image,sigma)

        return image

    def random_brightness(self,image, low, high, rnd=np.random.uniform):

        image = image+rnd(low,high)

        return image

    def random_contrast(self,image, low, high, rnd=np.random.uniform):

        mean = np.mean(image)

        image = (image-mean)*rnd(low,high)+mean

        return image

    def random_gamma(self,image, low, high, rnd=np.random.uniform):

        min = np.min(image)

        image = (image-min)*rnd(low,high)+min

        return image

    def random_flip(self,image, labels, rnd=np.random.rand()):

        if rnd < 0.5:

            image[0,:,:,:] = np.fliplr(image[0,:,:,:])

            labels[0,:,:,:] = np.fliplr(labels[0,:,:,:])

        if rnd > 0.5:

            image[0,:,:,:] = np.flipud(image[0,:,:,:])

            labels[0,:,:,:] = np.flipud(labels[0,:,:,:])

        return image,labels

    def get_trasform(self):

        self.image = self.random_brightness(self.image,low = -1,high = 1)

        self.image = self.random_contrast(self.image,low = 0.5,high = 1.5)

        self.image = self.random_gamma(self.image,low = 0.5,high = 1.5)

        self.image,self.labels = self.random_flip(self.image,self.labels)

        self.image = self.random_blur(self.image,low = 0, high = 2)

        return self.image,self.labels

class R2_CHs(object):

    """
    * R2_CHs: class to calculate the regression metric R^2 between the predicted CHs
    and the true CHs for each chemical element. Input parameters:

    - batch_predictions: batch of predictions maps from the FCN (batch_size, 256,256,5).
    - batch_labels: batch of ground truth maps (batch_size, 256,256,5).
    - num_chemical_elements: number of chemical_elements and output channels of the predictions and labels.

    """

    def __init__(self,batch_predictions, batch_labels,num_chemical_elements):

        self.batch_predictions = np.array(batch_predictions)

        self.batch_labels = np.array(batch_labels)

        self.num_chemical_elements = num_chemical_elements

    def get_peaks_pos(self,labels):

        peaks_pos = peak_local_max(labels,min_distance = 1,threshold_abs = 1e-6)

        return peaks_pos


    def get_CHs(self,labels, peaks_pos):

        """
        * get_CHs: function to extract the CHs, which are the values of the outputs
        in correspondence of the peaks. Input parameters:

        - output: predictions or labels.
        - peaks_pos: pixel positions of the column's peaks.

        """

        CHs = np.round(labels[peaks_pos[:, 0],
                                  peaks_pos[:, 1]])

        return CHs

    def get_r2(self,predictions,labels,peaks_predictions):

        """
        * get_r2: function to calculate the R^2 between the predicted and true CHs for a single element
        Input parameters:

        - predictions: predicted CHs mao of a single element.
        - labels: true CHs map of a single element.
        - peaks_predictions: bool (True or False). If True, the R^2 is calculated in the peaks
          of the predictions, if False the R^2 is calculated in the peaks if the ground truth.

          True: taking into account the false positive.
          False: taking into account the true negative.

          The R^2 is calculated for both the cases and the final value is the average of the two
          If there are less than 2 columns in the prediction, or if the R^2 is negative, the value
          is set to 0.

        """

        if peaks_predictions:

            peaks_pos = self.get_peaks_pos(predictions)

        else:

            peaks_pos = self.get_peaks_pos(labels)

        if len(peaks_pos) > 2:

            CHs_predictions = self.get_CHs(predictions, peaks_pos)

            CHs_labels = self.get_CHs(labels, peaks_pos)

            r2 = r2_score(CHs_predictions,CHs_labels)

            if r2 < 0.0:

                r2 = 0.0
        else:

            r2 = 0.0

        return r2

    def get_avg_r2(self,predictions,labels):

        """
        * get_avg_r2: function to calculate the average of the R^2 between the case
        of CHs calculated in the predicted peaks and the true peaks.

        """

        r2_1 = self.get_r2(predictions,labels,peaks_predictions = True)
        r2_2 = self.get_r2(predictions,labels,peaks_predictions = False)

        avg_r2 = (r2_1 + r2_2)/2

        return avg_r2

    def get_r2_all_elements(self):

        """
        * get_r2_all_elements: function to calculate the R^2 for each element (each output channel)

        """

        r2_all_elements = []

        for i in range(self.num_chemical_elements):

            predictions_single_element = self.predictions[i,:,:]

            labels_single_element = self.labels[i,:,:]

            r2_single_element = self.get_avg_r2(predictions_single_element,
                                                labels_single_element)

            r2_all_elements.append(r2_single_element)

        return r2_all_elements

    def get_r2_all_elements_batch(self):

        """
        * get_r2_all_elements_batch: function to calculate the R^2 for each element
        and for each data in the batch. Once the R^2 is calculated for each element,
        an average R^2 among all the elements is considered.

        """

        r2_1 = 0.0
        r2_2 = 0.0
        r2_3 = 0.0
        r2_4 = 0.0
        r2_5 = 0.0

        num_images = self.batch_predictions.shape[0]

        for self.predictions, self.labels in zip(self.batch_predictions, self.batch_labels):

            r2_all_elements = self.get_r2_all_elements()

            r2_1 += r2_all_elements[0]
            r2_2 += r2_all_elements[1]
            r2_3 += r2_all_elements[2]
            r2_4 += r2_all_elements[3]
            r2_5 += r2_all_elements[4]


        r2_1 = r2_1/num_images
        r2_2 = r2_2/num_images
        r2_3 = r2_3/num_images
        r2_4 = r2_4/num_images
        r2_5 = r2_5/num_images

        average_r2 = (r2_1 + r2_2 + r2_3 + r2_4 + r2_5)/self.num_chemical_elements

        return r2_1,r2_2,r2_3,r2_4,r2_5,average_r2


def save_epoch_results(training_results_folder_path,
                       test_results_folder_path,
                       epoch,
                       save_every,
                       model,
                       train_loss_learning_curve,
                       train_r2_learning_curve,
                       test_loss_learning_curve,
                       test_r2_learning_curve,
                       train_loss,
                       r2_all_elements_train,
                       test_loss,
                       r2_all_elements_test):

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
                         path = os.path.split(os.path.split(training_results_folder_path)[0])[0])

    if epoch % save_every == 0:
        torch.save(model.state_dict(),os.path.join(training_results_folder_path, 'weights/epoch-{}.pkl'.format(epoch + 1)))


def plot_debug(batch_images,batch_labels,batch_predictions,debug_folder_path):

    chemical_elements = ['Pt','Ni','Pd','Co','Fe']

    ce = 1

    for i in range(len(batch_images)):


        fig = plt.figure(figsize=(21,7))
        ax = fig.add_subplot(1, 3, 1)

        im = ax.imshow(batch_images[i,0,:,:], cmap='gray')
        plt.title('STEM Image',fontsize = 20)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax1)

        ax = fig.add_subplot(1, 3, 2)
        im = ax.imshow(batch_labels[i,ce - 1,:,:], cmap='jet')
        plt.title('{} Ground Truth'.format(chemical_elements[ce - 1]), fontsize=20)
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax2)

        ax = fig.add_subplot(1, 3, 3)
        im = ax.imshow(batch_predictions[i, ce - 1, :, :], cmap='jet')
        plt.title('{} Prediction'.format(chemical_elements[ce - 1]), fontsize=20)
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax2)

        plt.tight_layout()

        fig.savefig(debug_folder_path +'data_{}.png'.format(i + 1), bbox_inches='tight')

        plt.close(fig)

def plot_learning_curves(train_loss,test_loss,train_r2,test_r2,path):

    epochs = np.arange(1,len(train_loss) + 1)

    fig = plt.figure(figsize=(20, 20))

    fig.add_subplot(2, 2, 1)
    plt.plot(epochs,train_loss,'bo-')
    plt.plot(epochs, test_loss, 'ro-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training','Test'])

    fig.add_subplot(2, 2, 2)
    plt.plot(epochs, train_r2[:,5], 'bo-')
    plt.plot(epochs, test_r2[:,5], 'ro-')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend(['Training', 'Test'])

    fig.add_subplot(2, 2, 3)
    plt.plot(epochs, train_r2[:, 0], 'gray')
    plt.plot(epochs, train_r2[:, 1], 'green')
    plt.plot(epochs, train_r2[:, 2], 'blue')
    plt.plot(epochs, train_r2[:, 3], 'pink')
    plt.plot(epochs, train_r2[:, 4], 'red')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend(['Pt', 'Ni','Pd','Co','Fe'])

    fig.add_subplot(2, 2, 4)
    plt.plot(epochs, test_r2[:, 0], 'gray')
    plt.plot(epochs, test_r2[:, 1], 'green')
    plt.plot(epochs, test_r2[:, 2], 'blue')
    plt.plot(epochs, test_r2[:, 3], 'pink')
    plt.plot(epochs, test_r2[:, 4], 'red')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend(['Pt', 'Ni', 'Pd', 'Co', 'Fe'])

    plt.tight_layout()

    fig.savefig(path + '/learning_curves.png', bbox_inches='tight')

    plt.close(fig)