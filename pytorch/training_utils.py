
import numpy as np
import torch

from skimage.feature import peak_local_max
from skimage.filters import gaussian

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class R2_CHs(object):

    """
    * R2_CHs: class to calculate the regression metric R^2 between the predicted CHs
    and the true CHs for each chemical element. Input parameters:

    - batch_predictions: batch of predictions maps from the FCN (batch_size, 256,256,5).
    - batch_labels: batch of ground truth maps (batch_size, 256,256,5).
    - num_chemical_elements: number of chemical_elements and output channels of the predictions and labels.

    """

    def __init__(self,batch_predictions, batch_labels,num_chemical_elements = 5):

        self.batch_predictions = batch_predictions

        self.batch_labels = batch_labels.detach().cpu().numpy()

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

    def random_crop(self,image,labels,size):

        orig_size = image.shape[1:3]

        if orig_size>size:

            n = np.random.randint(0,orig_size[0]-size[0])
            m = np.random.randint(0,orig_size[1]-size[1])


            image = image[0,n:n+size[0],m:m+size[1],:]

            labels = labels[0,n:n+size[0],m:m+size[1],:]

        image, labels = image.reshape((1,)+image.shape),labels.reshape((1,)+labels.shape)

        return image, labels

    def get_trasform(self):

        self.image = self.random_brightness(self.image,low = -1,high = 1)

        self.image = self.random_contrast(self.image,low = 0.5,high = 1.5)

        self.image = self.random_gamma(self.image,low = 0.5,high = 1.5)

        self.image,self.labels = self.random_flip(self.image,self.labels)

        self.image = self.random_blur(self.image,low = 0, high = 2)

        return self.image,self.labels


def plot_debug(image,labels,i,debug_folder_path):

    """
    * plot_debug: function to save the plot of the input images and labels in debug folder
    (just to check what is going through the FCN). Input parameters:

    - image: input image.
    - labels: input labels.
    - i: index of the image in the batch.
    - debug_folder_path: path to debug folder.

    """

    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(1, 2, 1)

    im = ax.imshow(image[0,0,:,:], cmap='gray')
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax = cax1)

    ax = fig.add_subplot(1, 2, 2)
    im = ax.imshow(labels[0,0,:,:], cmap='jet')
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax = cax2)

    plt.tight_layout()

    fig.savefig(debug_folder_path +'data_{}.png'.format(i + 1), bbox_inches='tight')

    plt.close(fig)
