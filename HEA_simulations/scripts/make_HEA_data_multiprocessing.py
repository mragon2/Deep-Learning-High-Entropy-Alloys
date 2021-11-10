import numpy as np
import random

from make_HEA_data_utils import*

from ase.visualize import view

from pyqstem import PyQSTEM

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import platform
import multiprocessing as mp



print("Running on host '{}'".format(platform.node()))

n_processors_all = mp.cpu_count()
n_processors = 4
#n_processors = n_processors_all

print("Number of available processors: ", n_processors_all)
print("Number of used processors: ", n_processors)


# training
data_path = 'PtNiPdCoFe_data/training_data/'

first_number = 1
num_data = 8000

# test
# data_path = 'PtNiPdCoFe_data/test_data/'
#first_number = 1
# num_data_all = 2000

make_folder(data_path)
print('Saving simulated HEA in the directory {}'.format(data_path))

ATK_structure = False
crystal_structure = 'fcc'

ATK_structure = False
ATK_path = 'ATK_structures/'

# size of the crystal structure
low_size = 28
up_size = 35

# spatial domain represented in the image
# spatial_domain = (51.2, 51.2) #A
spatial_domain = (50, 50)  # A

# flag to impose a random translation of the HEA NP in the x,y directions
random_transl = True
t_xy = 1 / 8
random_rot = True
a_y = [0, 45, 90, 135, 180, 225, 270, 315]

# chemical symbols of the considered elements
chemical_symbols = ['Pt', 'Ni', 'Pd', 'Co', 'Fe']

# set the composition of each element "manually"
low_comp_Pt = ...
high_comp_Pt = ...

low_comp_Ni = ...
high_comp_Ni = ...

low_comp_Pd = ...
high_comp_Pd = ...

low_comp_Co = ...
high_comp_Co = ...

low_comp_Fe = ...
high_comp_Fe = ...

# set the composition of each element "randomly"
comp = 'random'

low_comp = 18  # lower bound of element composition
high_comp = 20  # upper bound of element composition

assert low_comp <= 100 / len(chemical_symbols), 'The minimal fraction of a single chemical element in a {} element HEA can not be higher than {:.0f}%, received {}% '.format(len(chemical_symbols),
                                                                                                                                                                            100 / len(chemical_symbols),
                                                                                                                                                                            low_f)

qstem = PyQSTEM('STEM')

image_size = (256, 256)  # px

resolution = spatial_domain[0] / image_size[0]  # [A/px]

probe = 8  # probe size [nm]

slice_thickness = 0.2  # [nm]

add_noise = False # depending if we want to add noise in the simulated STEM image

noise_mean = 0.0 # noise distribution mean (noise added is add_noise = True)

noise_std = 1.0  # noise distribution std (noise added is add_noise = True)

spot_size = 0.4

if ATK_structure:

    print('Creating random HEA models from VTK xyz file..')

else:

    print('Creating random HEA models from Random_Cluster..')

def HEA_multiprocessing(data_index):

    comp = 'random'

    print('Processing HEA [{}/{}]'.format(data_index, num_data))

    if ATK_structure:

        structure_path = os.path.join(ATK_path, '/structure_{}.xyz'.format(3))

        print('Creating random HEA model from VTK xyz file: {}..'.format(structure_path))

        random_HEA = ATK_Random_HEA(structure_path,
                                    spatial_domain,
                                    random_transl,
                                    t_xy,
                                    random_rot,
                                    a_y)

        random_HEA_model = random_HEA.get_model()

        cs = np.array(random_HEA_model.get_chemical_symbols())

        Pt_idx = np.where(cs == 'Pt')[0]
        Ni_idx = np.where(cs == 'Ni')[0]
        Pd_idx = np.where(cs == 'Pd')[0]
        Co_idx = np.where(cs == 'Co')[0]
        Fe_idx = np.where(cs == 'Fe')[0]

        cs_list = list(cs[Pt_idx]) + list(cs[Ni_idx] + list(Pd_idx) + list(Co_idx) + list(Fe_idx))
        random.shuffle(cs_list)

        cs_idx = len(Pt_idx)
        cs[Pt_idx] = cs_list[:cs_idx]

        cs[Ni_idx] = cs_list[cs_idx:cs_idx + len(Ni_idx)]
        cs_idx = cs_idx + len(Ni_idx)

        cs[Pd_idx] = cs_list[cs_idx:cs_idx + len(Pd_idx)]
        cs_idx = cs_idx + len(Pd_idx)

        cs[Co_idx] = cs_list[cs_idx:cs_idx + len(Co_idx)]
        cs_idx = cs_idx + len(Co_idx)

        cs[Fe_idx] = cs_list[cs_idx:cs_idx + len(Fe_idx)]
        cs_idx = cs_idx + len(Fe_idx)

        random_HEA_model.set_chemical_symbols(cs)

    else:

        random_size = np.random.uniform(low_size, up_size)  # A

        if comp is not 'random':

            comp_Pt = np.random.uniform(low_comp_Pt, high_comp_Pt)  # composition element 1
            comp_Ni = np.random.uniform(low_comp_Ni, high_comp_Ni)  # composition element 2
            comp_Pd = np.random.uniform(low_comp_Pd, high_comp_Pd)  # composition element 3
            comp_Co = np.random.uniform(low_comp_Co, high_comp_Co)  # composition element 4
            comp_Fe = 100 - comp_Pt - comp_Ni - comp_Pd - comp_Co  # composition element 5

            comp = [comp_Pt, comp_Ni, comp_Pd, comp_Co, comp_Fe]

            assert len(chemical_symbols) == len(comp), '{} fractions are required for {} chemical symbols'.format(
                len(chemical_symbols), len(chemical_symbols))

        else:

            random_HEA = Random_HEA(crystal_structure,
                                    random_size,
                                    spatial_domain,
                                    random_transl,
                                    t_xy,
                                    random_rot,
                                    a_y,
                                    chemical_symbols,
                                    comp,
                                    low_comp,
                                    high_comp)

        # generating the HEA model using the method 'get_model' of the Random_HEA class
        random_HEA_model = random_HEA.get_model()

    random_v0 = np.random.uniform(180, 220)  # acceleration voltage [keV]

    random_alpha = np.random.uniform(15, 20)  # convergence_angle [mrad]

    random_defocus = np.random.uniform(-10, 10)  # defocus [A]

    random_Cs = np.random.uniform(180, 220, size=1)[0]  # 1st order aberration

    random_astig_mag = np.random.uniform(18, 22)  # astigmation magnitude [A]

    random_astig_angle = np.random.uniform(12, 16)  # astigmation angle [A]

    random_aberrations = {'a33': 0, 'phi33': 30}

    HEA_stem = HEA_STEM(qstem,
                        random_HEA_model,
                        image_size,
                        resolution,
                        probe,
                        slice_thickness,
                        random_v0,
                        random_alpha,
                        random_defocus,
                        random_Cs,
                        random_astig_mag,
                        random_astig_angle,
                        random_aberrations,
                        add_noise,
                        noise_mean,
                        noise_std)

    img = HEA_stem.get_HEA_stem()

    HEA_labels = HEA_Labels(random_HEA_model,
                            image_size,
                            resolution,
                            spot_size)

    lbl = HEA_labels.get_labels_multi_elements()

    HEA_data = HEA_Data(random_HEA_model,
                        img,
                        lbl,
                        data_path,
                        data_index)

    HEA_data.save_HEA()


if __name__ == '__main__':

    for counter in range(0,num_data // n_processors):

        first_number = n_processors * counter + 1

        pool = mp.Pool(n_processors)
        pool.map(HEA_multiprocessing, [data_index for data_index in range(first_number,n_processors + first_number)])
        pool.close()

    #pool = mp.Pool(n_processors)
   # pool.map(HEA_multiprocessing, [data_index for data_index in range(first_number,n_processes + first_number)])
    #pool.close()