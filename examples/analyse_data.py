import uproot                                                                                       # importing all usefull packages
import corsika_toy_iact.iact_array as iact
import numpy as np
import pkg_resources
import matplotlib.pyplot as plt
from ctapipe.image import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
import time

data_dir  = '/Users/julianrypalla/PycharmProjects/CORSIKA_toy_IACT/corsika_toy_iact/data/'
save_dir = '/Users/julianrypalla/PycharmProjects/CORSIKA_toy_IACT/corsika_toy_iact/save_dir/'
intensity, width, length, impact, header_energy = None, None, None, None, None #priming the arrays


for i in range (1,2): #201
    if i < 10:
        data_i = 'CER00000' + str(i) + '.image.npz'
    elif i > 9 and i < 100:
        data_i = 'CER0000' + str(i) + '.image.npz'
    elif i > 99:
        data_i = 'CER000' + str(i) + '.image.npz'
    elif i > 201 : break

    dataset = "gammas_0deg/"+data_i #"EPOS_10000GeV_proton.root"
    filepath = data_dir+dataset
    print('Now analyzing dataset', data_i)

    # setting up the IACT array
    if 'proton' in dataset: #proton data
        positions = [[0,0],[20,0],[40,0],[60,0],[80,0],[100,0],[120,0],[140,0],[160,0],[180,0],[200,0],[220,0],[240,0],[260,0],[280,0],[300,0],[320,0],[340,0],[360,0],[380,0],[400,0]]
        iact_array = iact.IACTArray(positions, radius=6, multiple_cores=False)
    if '0deg' in dataset: #gamma dataset
        x = np.linspace(-120, 120, 3)
        xx, yy = np.array(np.meshgrid(x, x))
        positions = np.vstack((xx.ravel(), yy.ravel())).T
        iact_array = iact.IACTArray(positions, radius=6)
    else: #Corsika .root early test gammas
        x = np.linspace(-120, 120, 3)
        xx, yy = np.array(np.meshgrid(x, x))
        positions = np.vstack((xx.ravel(), yy.ravel())).T
        iact_array = iact.IACTArray(positions, radius=6, multiple_cores=True)

    telescope_x = np.array([pos[0] for pos in positions])
    telescope_y = np.array([pos[1] for pos in positions])

    # processing image
    if '0deg' in dataset: # Create new images
        loaded = np.load(filepath)
        images_loaded = loaded["images"].astype("float32")
        header = loaded["header"]
        iact_array.images = images_loaded

    else:
        header, image = iact_array.process_file_list(filepath)

    print('starting HP calc')
    starttime = time.time()
    image_pe = iact_array.scale_to_photoelectrons(psf_width=0.04, mirror_reflectivity=0.8, quantum_efficiency=0.2, pedestal_width=0.8, single_pe_width=0.5, photon_cut=10)
    geometry = iact_array.get_camera_geometry()
    i, j = 0, 0
    intensity_event = np.zeros((image_pe.shape[0], image_pe.shape[1]))
    width_event = np.zeros((image_pe.shape[0], image_pe.shape[1]))
    length_event = np.zeros((image_pe.shape[0], image_pe.shape[1]))
    print('starting loop')
    for event in image_pe:
        for image in event:
            mask = tailcuts_clean(geometry, image.ravel(),picture_thresh=8, boundary_thresh=4).reshape(40, 40)
            image[np.invert(mask)] = 0
            try:
                hill = hillas_parameters(geometry, image.ravel())
                intensity_event[i][j] = hill.intensity
                width_event[i][j] = hill.width.value
                length_event[i][j] = hill.length.value
                j += 1
            except HillasParameterizationError:
                j += 1

        if intensity is None:
            intensity = intensity_event
            width = width_event
            length = length_event
        else:
            intensity = np.concatenate((intensity, intensity_event))
            width = np.concatenate((width, width_event))
            length = np.concatenate((length, length_event))

        i += 1
        j = 0
        #print(i)
        if i ==1000:print('needed',time.time()-starttime,'seconds for the first 1k images')
        if i==5000:print('Half way done')

    endtime = time.time()
    print('finished HP calc in', endtime-starttime,'seconds,now doing impact dist')

    # calculate impact distance
    if "proton" in dataset:
        pass
    if '0deg' in dataset: # Create new images
        impact_val = np.sqrt((header.T[5][:, np.newaxis] - telescope_x)**2 + (header.T[6][:, np.newaxis] - telescope_y)**2) #header_loaded.T[5] is core_x
        header_energy_val = header.T[1]
    else:
        impact_val = np.sqrt((header['core_x'][:, np.newaxis] - telescope_x) ** 2 + (header['core_y'][:, np.newaxis] - telescope_y) ** 2)
        header_energy_val = header['energy']

    impact = np.concatenate((impact, impact_val))
    header_energy = np.concatenate((header_energy, header_energy_val))


def make_hist(selection, bins=100, weights=False, squared=False):
    if selection == 'intensity':
        sel = intensity_selection
    elif selection == 'width':
        sel = width_selection
    elif selection == 'length':
        sel = length_selection
    else:
        print('false selection chosen')

    xaxis = impact.ravel()[sel]
    yaxis = np.log10(intensity.T.ravel()[sel])
    if weights == False:
        weight = None
    else:
        if squared == False:
            if selection == 'intensity':
                weight = energy_weight[sel]
            elif selection == 'width':
                weight = width.T.ravel()[width_selection]
            elif selection == 'length':
                weight = length.T.ravel()[length_selection]
        elif squared == True:
            if selection == 'intensity':
                weight = (energy_weight[sel] ** 2)
                print('No need to calculate intensity^2 as we dont calc std of energy')
            elif selection == 'width':
                weight = (width.T.ravel()[width_selection] ** 2)
            elif selection == 'length':
                weight = (length.T.ravel()[length_selection] ** 2)
    return plt.hist2d(xaxis, yaxis, bins=bins, weights=weight)

make_hist('width',bins=100, weights=False, squared=False)