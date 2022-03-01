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
times = []

for k in range (1,101): #101 for proton
    if k < 10:
        data_k = 'CER00000' + str(k) + '.image.npz'
    elif k > 9 and k < 100:
        data_k = 'CER0000' + str(k) + '.image.npz'
    elif k > 99:
        data_k = 'CER000' + str(k) + '.image.npz'
    elif k > 101 : break

    datapath = "EPOS_100GeV_p/" #"gammas_0deg/" #for gamma files
    dataset = datapath+data_k
    filepath = data_dir+dataset
    print('Now analyzing', data_k)

    # setting up the IACT array
    if 'proton' in datapath or 'p' in datapath: #proton data
        positions = [[0,0],[40,0],[80,0],[120,0],[160,0],[200,0],[240,0],[280,0],[320,0],[360,0],[400,0]]
        iact_array = iact.IACTArray(positions, radius=6, multiple_cores=False)
    elif '0deg' in datapath: #gamma dataset
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
    if '0deg' in datapath or 'p' in datapath: # Create new images
        loaded = np.load(filepath)
        images_loaded = loaded["images"].astype("float32")
        header = loaded["header"]
        iact_array.images = images_loaded
    else:
        header, image = iact_array.process_file_list(filepath)

    #print('starting HP calc')
    starttime = time.time()
    image_pe = iact_array.scale_to_photoelectrons(psf_width=0.04, mirror_reflectivity=0.8, quantum_efficiency=0.2, pedestal_width=0.8, single_pe_width=0.5, photon_cut=10)
    geometry = iact_array.get_camera_geometry()
    i, j = 0, 0
    intensity_event = np.zeros((image_pe.shape[0], image_pe.shape[1]))
    width_event = np.zeros((image_pe.shape[0], image_pe.shape[1]))
    length_event = np.zeros((image_pe.shape[0], image_pe.shape[1]))
    for event in image_pe:
        for image in event:
            mask = tailcuts_clean(geometry, image.ravel(),picture_thresh=8, boundary_thresh=4).reshape(40, 40)
            image[np.invert(mask)] = 0
            try:
                hill = hillas_parameters(geometry, image.ravel())
                if hill.intensity > 60:                                   #filtering only images with intensity above 60
                    intensity_event[i][j] = hill.intensity
                    width_event[i][j] = hill.width.value
                    length_event[i][j] = hill.length.value
                    j += 1
                else:
                    j+=1
            except HillasParameterizationError:
                j += 1
        i += 1
        j = 0

    # calculate impact distance
    if "p" in datapath:
        header_energy_val = header.T[1] #array of size (1000,) that's full of 1000 (GeV)
    elif '0deg' in datapath: # Create new images
        impact_val = np.sqrt((header.T[5][:, np.newaxis] - telescope_x)**2 + (header.T[6][:, np.newaxis] - telescope_y)**2) #header.T[5] is core_x
        header_energy_val = header.T[1]
    else:
        impact_val = np.sqrt((header['core_x'][:, np.newaxis] - telescope_x) ** 2 + (header['core_y'][:, np.newaxis] - telescope_y) ** 2)
        header_energy_val = header['energy']


    if intensity is None:
        intensity = intensity_event
        width = width_event
        length = length_event
        header_energy = header_energy_val
        #impact = impact_val
        if "p" in dataset:
            impact = impact
        else:
            impact = impact_val

    else:
        intensity = np.concatenate((intensity, intensity_event))
        width = np.concatenate((width, width_event))
        length = np.concatenate((length, length_event))
        header_energy = np.concatenate((header_energy, header_energy_val))
        #impact = np.concatenate((impact, impact_val))
        if "p" in dataset:
            pass
        else:
            impact = np.concatenate((impact, impact_val))


    endtime = time.time()
    timeinterval = endtime-starttime
    times = np.append(times, timeinterval)
    print('HP calculation took', np.around(timeinterval,2),'seconds')




def make_hist(selection, bins=100, weights=False, squared=False):
    if selection == 'intensity':
        sel = intensity_selection
    elif selection == 'width':
        sel = width_selection
    elif selection == 'length':
        sel = length_selection
    else:
        print('false selection chosen')

    xaxis = impact[sel]
    yaxis = np.log10(intensity[sel])
    if weights == False:
        weight = None
    else:
        if squared == False:
            if selection == 'intensity':
                weight = energy_weight[sel.ravel()]
            elif selection == 'width':
                weight = width[width_selection]
            elif selection == 'length':
                weight = length[length_selection]
        elif squared == True:
            if selection == 'intensity':
                weight = (energy_weight[sel.ravel()] ** 2)
                #print('No need to calculate intensity^2 as we dont calc std of energy')
            elif selection == 'width':
                weight = (width[width_selection] ** 2)
            elif selection == 'length':
                weight = (length[length_selection] ** 2)
    return plt.hist2d(xaxis, yaxis, bins=bins, weights=weight)


intensity_selection = intensity != 0
length_selection = length != 0
width_selection = width != 0
energy_weight = np.repeat(header_energy, 9)

#make_hist('width',bins=100, weights=False, squared=False)

#hists = []
#for type in ('intensity', 'width', 'length'):
#    for weighted in (True, False):
#        if weighted == True:
#           for squaring in (True, False):
#               hists = np.append(hists, make_hist(type, bins=100,weights=weighted,squared=squaring)[0].ravel())
#        else:
#           hists = np.append(hists, make_hist(type, bins=100,weights=weighted,squared=squaring)[0].ravel())
# What I probably need to do is to convert this array to a dictionary instead, but f**k that for now
x_coord_pos = []
for i in range(0,len(positions)):
    x_coord_pos.append(positions[i][0])

print('Now writing:' + "EPOS_100GeV_new.txt")
textfile = open(save_dir + "EPOS_100GeV_new.txt", "w+")
textfile.write('#intensity \n')
np.savetxt(textfile, intensity.ravel())
textfile.write('#length \n')
np.savetxt(textfile, length.ravel())
textfile.write('#width \n')
np.savetxt(textfile, width.ravel())
textfile.write('#energy \n')
np.savetxt(textfile, header_energy)
if "p" in datapath:
    textfile.write('#positions \n')
    np.savetxt(textfile, x_coord_pos)
    textfile.close()
else:
    textfile.write('#impact_dist \n')
    np.savetxt(textfile, impact.ravel())
    textfile.close()
