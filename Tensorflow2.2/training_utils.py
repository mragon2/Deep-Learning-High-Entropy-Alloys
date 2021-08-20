import numpy as np
import os

import tensorflow as tf

from skimage.feature import peak_local_max
from skimage.filters import gaussian

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import logging
import platform


def print_devices(num_devices):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logging.getLogger('tensorflow').setLevel(logging.FATAL)

    print("Running on host '{}'".format(platform.node()))

    if num_devices == 1:

        print('Running on {} device'.format(num_devices))

    else:

        print('Running on {} devices'.format(num_devices))



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

    num_data = len(os.listdir(path))

    data_path = os.path.join(path,str('*.npy'))

    dataset = tf.data.Dataset.list_files(data_path)

    dataset = dataset.shuffle(buffer_size = num_data).batch(batch_size = batch_size)

    num_batches = num_data // batch_size

    return dataset,num_batches



def make_batch(batch):

    batch = np.array(batch)

    batch_images = []

    batch_labels = []

    for i in range(len(batch)):

        data = np.load(batch[i])

        img = data[:,:,:,0]
        img = img.reshape(img.shape+(1,)).astype(np.float32)

        lbl = data[:,:,:,1:]

        rnd_imgng = Random_Imaging(image = img,labels = lbl)
        img,lbl = rnd_imgng.get_trasform()

        batch_images.append(img)

        batch_labels.append(lbl)

    batch_images = np.concatenate(batch_images)

    batch_labels = np.concatenate(batch_labels)

    return  [batch_images, batch_labels]

def make_model(FCN,input_shape, output_channels):

    input_channel = 1

    input_tensor = tf.keras.Input(shape = input_shape+(input_channel,))

    model = FCN(input_tensor, output_channels)

    return model

def compute_loss(labels, predictions,loss_object,batch_size):

    per_example_loss = loss_object(labels, predictions)

    per_example_loss /= tf.cast(tf.reduce_prod(tf.shape(labels)[1:]), tf.float32)

    return tf.nn.compute_average_loss(per_example_loss, global_batch_size = batch_size)


def make_config():

    config_proto = tf.compat.v1.ConfigProto()
    config_proto.allow_soft_placement = True
    config_proto.log_device_placement = False
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.force_gpu_compatible = True
    config_proto.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

def make_config_hvd(hvd):

    config_proto = tf.compat.v1.ConfigProto()
    config_proto.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.compat.v1.InteractiveSession(config = config_proto)

@tf.function
def train_step(batch,model,mp,loss_object,optimizer):

    batch_images, batch_labels = batch

    batch_size = len(batch_images)

    with tf.GradientTape() as tape:

        batch_predictions = model(batch_images, training=True)

        train_loss = compute_loss(batch_labels, batch_predictions, loss_object, batch_size)

        if mp:
            scaled_train_loss = optimizer.get_scaled_loss(train_loss)

    if mp:

        scaled_gradients = tape.gradient(scaled_train_loss, model.trainable_variables)

        gradients = optimizer.get_unscaled_gradients(scaled_gradients)

    else:

        gradients = tape.gradient(train_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return train_loss, batch_predictions

@tf.function
def distributed_train_step(strategy,global_batch,model,mp,loss_object,optimizer):

    per_replica_losses, per_replica_predictions = strategy.run(train_step, args=(global_batch,model,mp,loss_object,optimizer,))

    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)

    predictions = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_predictions,axis=None)

    return loss,predictions


@tf.function
def train_step_hvd(batch,model,mp,loss_object,optimizer,hvd,first_batch):

    batch_images, batch_labels = batch

    with tf.GradientTape() as tape:

        batch_predictions = model(batch_images, training = True)

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

    return train_loss,batch_predictions


@tf.function
def test_step(batch,model,loss_object):

    batch_images, batch_labels = batch

    batch_size = len(batch_images)

    batch_predictions = model(batch_images, training = True)

    test_loss = compute_loss(batch_labels, batch_predictions, loss_object, batch_size)

    return test_loss, batch_predictions


@tf.function
def distributed_test_step(strategy,global_batch,model,loss_object):

    per_replica_losses, per_replica_predictions = strategy.run(test_step, args=(global_batch,model,loss_object,))

    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    predictions = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_predictions, axis=None)

    return loss, predictions

@tf.function
def test_step_hvd(batch,model,loss_object,optimizer,hvd,first_batch):

    batch_images, batch_labels = batch

    batch_predictions = model(batch_images, training = True)

    test_loss = loss_object(batch_labels, batch_predictions)

    if first_batch:

        hvd.broadcast_variables(model.variables, root_rank=0)

        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    return test_loss,batch_predictions


class R2_CHs(object):

    """
    * R2_CHs: class to calculate the regression metric R^2 between the predicted CHs
    and the true CHs for each chemical element. Input parameters:

    - batch_predictions: batch of predictions maps from the FCN (batch_size, 256,256,5).
    - batch_labels: batch of ground truth maps (batch_size, 256,256,5).
    - num_chemical_elements: number of chemical_elements and output channels of the predictions and labels.

    """

    def __init__(self,batch_predictions,batch_labels,num_chemical_elements = 5):

        self.batch_predictions = batch_predictions.numpy()

        self.batch_labels = batch_labels

        self.num_chemical_elements = num_chemical_elements

    def get_peaks_pos(self,labels):

        peaks_pos = peak_local_max(labels,min_distance = 1,threshold_abs = 1e-6)

        return peaks_pos


    def get_CHs(self,output, peaks_pos):

        """
        * get_CHs: function to extract the CHs, which are the values of the outputs
        in correspondence of the peaks. Input parameters:

        - output: predictions or labels.
        - peaks_pos: pixel positions of the column's peaks.

        """

        CHs = np.round(output[peaks_pos[:, 0],
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

            predictions_single_element = self.predictions[:,:,i]

            labels_single_element = self.labels[:,:,i]

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

        for self.predictions,self.labels in zip(self.batch_predictions,self.batch_labels):


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


    def get_trasform(self):

        self.image = self.random_brightness(self.image,low = -1,high = 1)

        self.image = self.random_contrast(self.image,low = 0.5,high = 1.5)

        self.image = self.random_gamma(self.image,low = 0.5,high = 1.5)

        self.image,self.labels = self.random_flip(self.image,self.labels)

        self.image = self.random_blur(self.image,low = 0, high = 2)

        return self.image,self.labels

def plot_debug(batch_images,batch_labels,batch_predictions,debug_folder_path):

    chemical_elements = ['Pt','Ni','Pd','Co','Fe']

    ce = 1

    for i in range(len(batch_images)):


        fig = plt.figure(figsize=(21,7))
        ax = fig.add_subplot(1, 3, 1)

        im = ax.imshow(batch_images[i,:,:,0], cmap='gray')
        plt.title('STEM Image',fontsize = 20)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax1)

        ax = fig.add_subplot(1, 3, 2)
        im = ax.imshow(batch_labels[i,:,:,ce - 1], cmap='jet')
        plt.title('{} Ground Truth'.format(chemical_elements[ce - 1]), fontsize=20)
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax2)

        ax = fig.add_subplot(1, 3, 3)
        im = ax.imshow(batch_predictions[i, :, :, ce - 1], cmap='jet')
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

    fig.savefig(path + 'learning_curves.png', bbox_inches='tight')

    plt.close(fig)

