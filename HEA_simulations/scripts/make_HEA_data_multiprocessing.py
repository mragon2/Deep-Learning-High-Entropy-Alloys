import numpy as np

from make_HEA_data_utils import Random_HEA, HEA_STEM, HEA_Labels, HEA_Data

from ase.visualize import view

from pyqstem import PyQSTEM

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import platform
import multiprocessing as mp



print("Running on host '{}'".format(platform.node()))

n_processors = mp.cpu_count()
num_data = n_processors
print("Number of available processors: ", n_processors)
print("Number of used processors: ", num_data)


chemical_symbols = ['Pt', 'Ni', 'Pd', 'Co', 'Fe']

# training
path = 'PtNiPdCoFe_data/training_data/'
num_data_all = 8000

# test
#path = 'PtNiPdCoFe_data/training_data/'
#num_data_all = 2000


crystal_structure = 'fcc'

low_size = 30
up_size = 40

spatial_domain = (51.2, 51.2)  # A

random_transl = False

chemical_symbols = ['Pt', 'Ni', 'Pd', 'Co', 'Fe']

low_comp = 15  # lower bound of element composition
high_comp = 40  # upper bound of element composition

assert low_comp <= 100 / len(chemical_symbols), 'The minimal fraction of a single chemical element in a {} element HEA can not be higher than {:.0f}%, received {}% '.format(len(chemical_symbols),
                                                                                                                                                                            100/len(chemical_symbols),
                                                                                                                                                                             low_f)
qstem = PyQSTEM('STEM')

image_size = (256, 256)  # px

resolution = spatial_domain[0] / image_size[0]  # [A/px]

probe = 8  # probe size [nm]

slice_thickness = 0.2  # [nm]

add_noise = False # depending if we want to add noise in the simulated STEM image

noise_mean = 0.0 # noise distribution mean (noise added is add_noise = True)

noise_std = 1.0  # noise distribution std (noise added is add_noise = True)



def HEA_multiprocessing(data_index):

    print('Processing HEA [{}/{}]'.format(data_index, num_data_all))

    random_size = np.random.uniform(low_size, up_size)  # A

    comp_1 = np.random.uniform(5, 20)  # composition element 1
    comp_2 = np.random.uniform(5, 20)  # composition element 2
    comp_3 = np.random.uniform(5, 20)  # composition element 3
    comp_4 = np.random.uniform(5, 20)  # composition element 4
    comp_5 = 100 - comp_1 - comp_2 - comp_3 - comp_4 # composition element 5
    comp = [comp_1,comp_2,comp_3,comp_4,comp_5]

    #comp = None

    assert len(chemical_symbols) == len(comp), '{} fractions are required for {} chemical symbols'.format(len(chemical_symbols),
                                                                                                        len(chemical_symbols))

    random_HEA = Random_HEA(crystal_structure,
                            random_size,
                            spatial_domain,
                            random_transl,
                            chemical_symbols,
                            comp,
                            low_comp,
                            high_comp)

    random_HEA_model = random_HEA.get_model()

    cs = random_HEA_model.get_chemical_symbols()

    #print('Concentration of nanoparticle [{}/{}]'.format(data_index, num_data_all))
    #print('% of Pt = {:.2f}'.format(len(np.where(np.array(cs) == 'Pt')[0])/len(cs)))
    #print('% of Ni = {:.2f}'.format(len(np.where(np.array(cs) == 'Ni')[0])/len(cs)))
    #print('% of Pd = {:.2f}'.format(len(np.where(np.array(cs) == 'Pd')[0])/len(cs)))
    #print('% of Co = {:.2f}'.format(len(np.where(np.array(cs) == 'Co')[0])/len(cs)))
    #print('% of Fe = {:.2f}'.format(len(np.where(np.array(cs) == 'Fe')[0])/len(cs)))
    #print('')

    random_v0 = np.random.uniform(180, 220)  # acceleration voltage [keV]

    random_alpha = np.random.uniform(15, 20)  # convergence_angle [mrad]

    random_defocus = np.random.uniform(-10, 10)  # defocus [A]

    random_Cs = np.random.uniform(180, 220)  # 1st order aberration

    random_astig_mag = np.random.uniform(18, 22)  # astigmation magnitude [A]

    random_astig_angle = np.random.uniform(12, 16)  # astigmation angle [A]

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
                        add_noise,
                        noise_mean,
                        noise_std)

    img = HEA_stem.get_HEA_stem()

    HEA_labels = HEA_Labels(random_HEA_model,
                            image_size,
                            resolution)

    lbl = HEA_labels.get_labels_multi_elements()

    HEA_data = HEA_Data(random_HEA_model,
                        img,
                        lbl,
                        path,
                        data_index)

    HEA_data.save_HEA()


if __name__ == '__main__':

    for counter in range(1,160):

        first_number = num_data * counter + 1

        pool = mp.Pool(n_processors)
        pool.map(HEA_multiprocessing, [data_index for data_index in range(first_number,num_data + first_number)])
        pool.close()

    pool = mp.Pool(n_processors)
    pool.map(HEA_multiprocessing, [data_index for data_index in range(first_number,num_data + first_number)])
    pool.close()
