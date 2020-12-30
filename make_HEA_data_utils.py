import numpy as np

import random

from ase.build import bulk
from ase import Atoms
from ase.io import write

from skimage.filters import gaussian
from skimage.feature import peak_local_max

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os


class Random_Cluster(object):

    directions = np.array(

        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0],
         [0, -1, 0], [0, 0, -1], [1, 1, 1], [-1, 1, 1],
         [1, -1, 1], [1, 1, -1], [-1, -1, 1], [-1, 1, -1],
         [1, -1, -1], [-1, -1, -1]]
    )

    directions = (directions.T / np.linalg.norm(directions, axis=1)).T

    def __init__(self,crystal_structure, random_size):

        self.crystal_structure = crystal_structure

        self.random_size = random_size

        self.sites = self.get_sites()

        self.bonds = self.get_bonds()

    def get_sites(self):

        grid_size = 22

        cluster = bulk('Pt', self.crystal_structure, a = 3.91, cubic = True)

        cluster *= (grid_size,) * 3

        cluster.center()

        self.center = np.diag(cluster.get_cell()) / 2

        positions = cluster.get_positions()

        return positions

    def get_bonds(self):

        bond_length = 3.91 / np.sqrt(2)

        bonds = []

        for i, s in enumerate(self.sites):

            distances = np.linalg.norm(self.sites - s, axis=1)

            indices = np.where(distances <= bond_length * 1.05)[0]

            bonds.append(indices)

        return bonds

    def create_seed(self, lengths100, lengths111):

        self.active = np.ones(len(self.sites), dtype=bool)

        lengths = np.hstack((lengths100, lengths111))

        for length, direction in zip(lengths, self.directions):

            r0 = self.center + length * direction

            for i, site in enumerate(self.sites):

                if self.active[i]:

                    self.active[i] = np.sign(np.dot(direction, site - r0)) == -1

        self.active_bonds = np.array([self.active[b] for b in self.bonds],dtype = object)

        self.available_sites = np.where([any(ab) & (not a) for ab, a in zip(self.active_bonds, self.active)])[0]

    def build(self, grid_size, T0, T1=None):

        if T1 is None:

            T1 = T0

        for i in range(grid_size):

            T = T0 + (T1 - T0) * i / grid_size

            coordination = self.get_coordination(self.available_sites)

            p = np.zeros_like(coordination, dtype=np.float)

            p[coordination > 2] = np.exp(coordination[coordination > 2] / T)

            p = p / float(np.sum(p))

            p[p < 0] = 0

            n = np.random.choice(len(p), p=p)

            k = self.available_sites[n]

            self.available_sites = np.delete(self.available_sites, n)

            self.expand(k)

    def expand(self, k):

        self.active[k] = True

        new_avail = self.bonds[k][self.active[self.bonds[k]] == 0]

        self.available_sites = np.array(list(set(np.append(self.available_sites, new_avail))))

        if len(new_avail) > 0:

            to_update = np.array([np.where(self.bonds[x] == k)[0] for x in new_avail]).T[0]

            for i, j in enumerate(to_update):

                self.active_bonds[new_avail][i][j] = True

    def get_coordination(self, sites):

        return np.array([sum(self.active_bonds[site]) for site in sites])

    def get_cluster(self):

        element = 'Pt'

        return Atoms([element] * len(self.sites[self.active]), self.sites[self.active])

    def get_model(self):

        radius = self.random_size/2

        lengths100 = np.random.uniform(radius, radius + .2 * radius, 6)

        lengths111 = np.random.uniform(radius, radius + .2 * radius, 8)

        self.create_seed(lengths100, lengths111)

        self.build(int(np.sum(self.active) / 4.), 10, 2)

        model = self.get_cluster()

        return model


class Random_HEA(object):
    
    def __init__(self,crystal_structure, random_size,spatial_domain,chemical_symbols):

        self.model = Random_Cluster(crystal_structure,random_size).get_model()

        self.spatial_domain = spatial_domain

        self.chemical_sysmbols = chemical_symbols


    def get_model(self):

        positions = self.model.get_positions()

        n_elements = len(self.chemical_sysmbols)

        random_pos_index = np.random.randint(1, n_elements + 1, size=len(positions))

        chemical_symbols_shuffled = []

        for i in range(len(positions)):

            random.shuffle(self.chemical_sysmbols)

            chemical_symbols_shuffled.append(self.chemical_sysmbols[random_pos_index[i] - 1])

        self.model.set_chemical_symbols(chemical_symbols_shuffled)

        self.model.rotate(v = 'y', a = 45, center='COP')

        self.model.rotate(v='z', a=np.random.random() * 360, center='COP')

        self.model.center(vacuum=0)

        cell = self.model.get_cell()

        size = np.diag(cell)

        self.model.set_cell((self.spatial_domain[0],) * 3)

        self.model.center()

        tx = np.random.randint(-self.spatial_domain[0] * 5 / 8, self.spatial_domain[0] * 5 / 8, size=1)[0]

        ty = np.random.randint(-self.spatial_domain[1] * 5 / 8, self.spatial_domain[1] * 5 / 8, size=1)[0]

        self.model.translate((tx, ty, 0))

        return self.model


class HEA_STEM(object):

    def __init__(self,qstem,HEA_model,image_size, resolution, probe, slice_thickness, v0, alpha, defocus, Cs, astig_mag, astig_angle, add_noise,noise_mean, noise_std):

        self.qstem = qstem

        self.HEA_model = HEA_model

        self.image_size = image_size

        self.resolution = resolution

        self.probe = probe

        self.slice_thickness = slice_thickness

        self.v0 = v0

        self.alpha = alpha

        self.defocus = defocus

        self.Cs = Cs

        self.astig_mag = astig_mag

        self.astig_angle = astig_angle

        self.add_noise = add_noise

        self.noise_mean = noise_mean

        self.noise_std = noise_std


    def get_DF_img(self):

        self.qstem.set_atoms(self.HEA_model)

        self.qstem.build_probe(
                               v0 = self.v0,
                               alpha = self.alpha,
                               num_samples = (self.probe, self.probe),
                               resolution = (self.resolution,self.resolution),
                               Cs = self.Cs,
                               defocus = self.defocus,
                               astig_mag = self.astig_mag,
                               astig_angle = self.astig_angle
                                )

        self.wave = self.qstem.get_wave()

        self.num_slices = self.probe / self.slice_thickness

        self.spatial_domain =  (self.image_size[0] * self.resolution, self.image_size[1] * self.resolution)

        self.scan_range = [[0, self.spatial_domain[0], self.image_size[0]],
                          [0, self.spatial_domain[1], self.image_size[1]]]

        self.qstem.build_potential(num_slices = self.num_slices, scan_range = self.scan_range)

        detector1_radii = (70, 200)

        detector2_radii = (0, 70)

        self.qstem.add_detector('detector1', detector1_radii)

        self.qstem.add_detector('detector2', detector2_radii)

        self.qstem.run()

        self.img = self.qstem.read_detector('detector2') * (-1)

        return self.img


    def get_local_normalization(self):

        self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))

        self.img = self.img - gaussian(self.img, 12 / self.resolution)

        self.img = self.img / np.sqrt(gaussian(self.img ** 2, 12 / self.resolution))

        return self.img


    def get_local_noise(self):

        self.peaks = peak_local_max(self.img, min_distance = 1, threshold_abs = 1e-6)

        self.distance = 12

        for _, p in enumerate(self.peaks):

            for i in range(p[0] - self.distance // 2, p[0] + self.distance // 2):

                for j in range(p[1] - self.distance // 2, p[1] + self.distance // 2):

                    if (i > 0 and j > 0 and i > 0 and j < self.image_size[1] and i < self.image_size[0] and j > 0 and i < self.image_size[0] and j < self.image_size[1]):

                        self.img[i][j] = self.img[i][j] + np.random.normal(self.noise_mean, self.noise_std, (1, 1))

        self.img[self.img <-1] = -1

        return self.img

    def get_HEA_stem(self):

        self.get_DF_img()

        self.get_local_normalization()

        if self.add_noise:

            self.get_local_noise()

        return self.img

class HEA_Labels(object):

    def __init__(self,HEA_model,image_size,resolution):

        self.HEA_model = HEA_model

        self.image_size = image_size

        self.resolution = resolution

    def get_labels_single_element(self,element):

        positions = self.HEA_model.get_positions()[np.array(self.HEA_model.get_chemical_symbols()) == element][:,:2]/self.resolution

        spotsize = 0.52

        width = int(spotsize/self.resolution)

        x, y = np.mgrid[0:self.image_size[0], 0:self.image_size[1]]

        labels = np.zeros(self.image_size)

        for p in (positions):

            p_round = np.round(p).astype(int)

            min_xi = np.max((p_round[0] - width * 4, 0))
            max_xi = np.min((p_round[0] + width * 4 + 1, self.image_size[0]))
            min_yi = np.max((p_round[1] - width * 4, 0))
            max_yi = np.min((p_round[1] + width * 4 + 1, self.image_size[1]))

            xi = x[min_xi:max_xi, min_yi:max_yi]
            yi = y[min_xi:max_xi, min_yi:max_yi]

            v = np.array([xi.ravel(), yi.ravel()])

            labels[xi, yi] += np.exp(-cdist([p], v.T) ** 2 / (2 * width ** 2)).reshape(xi.shape)

        return labels

    def get_labels_multi_elements(self):

        labels_all_elements = []

        for element in np.unique(self.HEA_model.get_chemical_symbols()):

            labels_single_element = self.get_labels_single_element(element)

            labels_single_element = labels_single_element.reshape(labels_single_element.shape + (1,))

            labels_all_elements.append(labels_single_element)

        labels_all_elements = np.concatenate(labels_all_elements, axis = 2)

        return labels_all_elements


class HEA_Data(object):

    def __init__(self,model,img,lbl,path,data_index):

        self.model = model

        self.img = img

        self.lbl = lbl

        self.path = path

        self.data_index = data_index

    def save_HEA_model(self):

        if(self.path + 'models/') and not os.path.exists(self.path + 'models/'):
            os.makedirs(self.path + 'models/')

        write(self.path + 'models/HEA_model_{}.xyz'.format(self.data_index), self.model)


    def save_HEA_data(self):

        self.img = self.img.reshape((1,) + self.img.shape + (1,))

        self.lbl = self.lbl.reshape((1,) + self.lbl.shape)

        self.data = np.concatenate([self.img,self.lbl], axis = 3)

        if (self.path + 'data/') and not os.path.exists(self.path + 'data/'):
            os.makedirs(self.path + 'data/')

        np.save(self.path + 'data/data_{}.npy'.format(self.data_index),self.data)

        return self.data

    def save_HEA_plot(self):

        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(1, 2, 1)

        im = ax.imshow(self.img[0, :, :, 0], cmap='gray')
        plt.title('STEM image', fontsize=20)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax1)

        ax = fig.add_subplot(1, 2, 2)
        im = ax.imshow(self.lbl[0, :, :, 0], cmap='jet')
        plt.title('Pt labels',fontsize = 20)
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax2)

        plt.tight_layout()

        if (self.path + 'plots/') and not os.path.exists(self.path + 'plots/'):
            os.makedirs(self.path + 'plots/')

        fig.savefig(self.path + 'plots/data_{}.png'.format(self.data_index), bbox_inches='tight')

        plt.close(fig)

    def save_HEA(self):

        self.save_HEA_model()

        self.save_HEA_data()

        self.save_HEA_plot()






















