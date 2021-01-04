import numpy as np

from make_HEA_data_utils import Random_HEA, HEA_STEM, HEA_Labels, HEA_Data

from ase.visualize import view

from pyqstem import PyQSTEM

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import platform



if __name__ == '__main__':

    path = 'training_data/'
    first_number = 1
    num_data_all = 8000

    # test
    #path = 'test_data/'
    #first_number = 1
    #num_data_all = 2000

    crystal_structure = 'fcc'

    spatial_domain = (51.2, 51.2)  # A

    chemical_symbols = ['Pt', 'Ni', 'Pd', 'Co', 'Fe']

    qstem = PyQSTEM('STEM')

    image_size = (256, 256)  # px

    resolution = spatial_domain[0] / image_size[0]  # [A/px]

    probe = 8  # probe size [nm]

    slice_thickness = 0.2  # [nm]

    add_noise = False # depending if we want to add noise in the simulated STEM image

    noise_mean = 0.0 # noise distribution mean (noise added is add_noise = True)

    noise_std = 1.0  # noise distribution std (noise added is add_noise = True)


    print("Running on host '{}'".format(platform.node()))

    for data_index in range(first_number, num_data_all):

        print('Processing Data [{}/{}]'.format(data_index, num_data_all))

        random_size = np.random.uniform(5,10, 1)[0]  # A

        random_HEA = Random_HEA(crystal_structure, random_size, spatial_domain, chemical_symbols)

        random_HEA_model = random_HEA.get_model()

        random_v0 = np.random.randint(180, 220, size=1)[0]  # acceleration voltage [keV]

        random_alpha = np.random.randint(15, 20, size=1)[0]  # convergence_angle [mrad]

        random_defocus = np.random.randint(-10, 10, size=1)[0]  # defocus [A]

        random_Cs = np.random.randint(180, 220, size=1)[0]  # 1st order aberration

        random_asti_mag = np.random.randint(18, 22, size=1)[0]  # astigmation magnitude [A]

        random_asti_angle = np.random.randint(12, 16, size=1)[0]  # astigmation angle [A]

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
                        random_asti_mag,
                        random_asti_angle,
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
