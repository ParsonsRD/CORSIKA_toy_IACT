import uproot                                                                                       # importing all usefull packages
import corsika_toy_iact.iact_array as iact
import numpy as np
import pkg_resources
import matplotlib.pyplot as plt
from ctapipe.image import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError

# accessing dataset & filepaths
#data_dir = pkg_resources.resource_filename('corsika_toy_iact', 'data/gammas_0deg/')
data_dir  = '/Users/julianrypalla/PycharmProjects/CORSIKA_toy_IACT/corsika_toy_iact/data/'
#save_dir = pkg_resources.resource_filename('corsika_toy_iact', 'save_dir/InterpolTest/')
save_dir = '/Users/julianrypalla/PycharmProjects/CORSIKA_toy_IACT/corsika_toy_iact/save_dir/'

#for i in range (1,10):
    #data_i = 'CER00000' + str(i) + '.image.npz'
#for i in range (10,100):
    #data_i ='CER0000' + str(i) + '.image.npz'
for i in range (100,201):
    data_i = 'CER000' + str(i) + '.image.npz'

    dataset = "gammas_0deg/"+data_i #"EPOS_10000GeV_proton.root"
    filepath = data_dir+dataset
    print('Now analyzing dataset', data_i)
    # setting up the IACT array
    if 'proton' in dataset:
        positions = [[0,0],[20,0],[40,0],[60,0],[80,0],[100,0],[120,0],[140,0],[160,0],[180,0],[200,0],[220,0],[240,0],[260,0],[280,0],[300,0],[320,0],[340,0],[360,0],[380,0],[400,0]]
        iact_array = iact.IACTArray(positions, radius=6, multiple_cores=False)
    if '0deg' in dataset: # Create our 3x3 array
        x = np.linspace(-120, 120, 3)
        xx, yy = np.array(np.meshgrid(x, x))
        positions = np.vstack((xx.ravel(), yy.ravel())).T
        iact_array = iact.IACTArray(positions, radius=6)
    else: #Corsika .root gammas
        x = np.linspace(-120, 120, 3)
        xx, yy = np.array(np.meshgrid(x, x))
        positions = np.vstack((xx.ravel(), yy.ravel())).T
        iact_array = iact.IACTArray(positions, radius=6, multiple_cores=True)

    # processing image
    if '0deg' in dataset: # Create new images
        loaded = np.load(filepath)
        images_loaded = loaded["images"].astype("float32")
        header = loaded["header"]
        iact_array.images = images_loaded

    else:
        header, image = iact_array.process_file_list(filepath)

    image_pe = iact_array.scale_to_photoelectrons(psf_width=0.04, mirror_reflectivity=0.8, quantum_efficiency=0.2, pedestal_width=0.8, single_pe_width=0.5, photon_cut=10)
    geometry = iact_array.get_camera_geometry()
    i, j = 0, 0
    intensity = np.zeros((image_pe.shape[0], image_pe.shape[1]))
    width = np.zeros((image_pe.shape[0], image_pe.shape[1]))
    length = np.zeros((image_pe.shape[0], image_pe.shape[1]))
    for event in image_pe:
        for image in event:
            mask = tailcuts_clean(geometry, image.ravel(),
                                  picture_thresh=8, boundary_thresh=4).reshape(40, 40)
            image[np.invert(mask)] = 0
            try:
                hill = hillas_parameters(geometry, image.ravel())
                intensity[i][j] = hill.intensity
                width[i][j] = hill.width.value
                length[i][j] = hill.length.value
                j += 1
            except HillasParameterizationError:
                j += 1

        i += 1
        j = 0

    # getting average & std values (into the right shape)
    intensity = intensity.T
    avg_intensity = np.zeros(intensity.shape[0])
    std_intensity = np.zeros(intensity.shape[0])
    percentile_intensity = np.zeros(intensity.shape[0])
    length = length.T
    avg_length = np.zeros(length.shape[0])
    std_length = np.zeros(length.shape[0])
    percentile_length = np.zeros(length.shape[0])
    width = width.T
    avg_width = np.zeros(width.shape[0])
    std_width = np.zeros(width.shape[0])
    percentile_width = np.zeros(width.shape[0])

    for tel in range(intensity.shape[0]):
        tel_intensity = intensity[tel]
        avg_intensity[tel] = np.average(tel_intensity[tel_intensity>0])
        std_intensity[tel] = np.std(tel_intensity[tel_intensity>0])
        percentile_intensity[tel] = np.percentile(tel_intensity[tel_intensity>0], 10)
    for tel in range(length.shape[0]):
        tel_length = length[tel]
        avg_length[tel] = np.average(tel_length[tel_length>0])
        std_length[tel] = np.std(tel_length[tel_length>0])
        percentile_length[tel] = np.percentile(tel_length[tel_length>0], 10)
    for tel in range(width.shape[0]):
        tel_width = width[tel]
        avg_width[tel] = np.average(tel_width[tel_width>0])
        std_width[tel] = np.std(tel_width[tel_width>0])
        percentile_width[tel] = np.percentile(tel_width[tel_width>0], 10)

    # calculate impact distance
    telescope_x = np.array([pos[0] for pos in positions])
    telescope_y = np.array([pos[1] for pos in positions])
    if "proton" in dataset:
        pass
    if '0deg' in dataset: # Create new images
        impact = np.sqrt((header.T[5][:, np.newaxis] - telescope_x)**2 + (header.T[6][:, np.newaxis] - telescope_y)**2) #header_loaded.T[5] is core_x
        header_energy = header.T[1]
    else:
        impact = np.sqrt((header['core_x'][:, np.newaxis] - telescope_x) ** 2 + (header['core_y'][:, np.newaxis] - telescope_y) ** 2)
        header_energy = header['energy']

    # plotting

    if "proton" in dataset:
        fig=plt.figure(figsize=(7,7))
        fig.suptitle('Average HP for ' + dataset)
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        ax.set_xlabel('telescope position in m')
        fig.text(0.02, 0.5, 'Hillas Parameters', ha='center', va='center', rotation='vertical')

        #ax1.errorbar(telescope_x, percentile_intensity, xerr=None, yerr = None, marker='x', ls='--', label='Intensity')
        #ax2.errorbar(telescope_x, percentile_length, xerr=None, yerr = None, marker='x', ls='--', label='Length')
        #ax3.errorbar(telescope_x, percentile_width, xerr=None, yerr = None, marker='x', ls='--',label='Width')
        ax1.errorbar(telescope_x, avg_intensity, xerr=None, yerr = None, marker='x', ls='--', label='avg intensity')
        ax2.errorbar(telescope_x, avg_length, xerr=None, yerr = None, marker='x', ls='--', label='avg length')
        ax3.errorbar(telescope_x, avg_width, xerr=None, yerr = None, marker='x', ls='--',label='avg width')

        # making the plot pretty & readable
        ax1.grid()
        ax1.legend()
        ax1.set_ylabel('Intensity in photoelectrons')
        #ax1.set_yscale("log")
        ax2.grid()
        ax2.legend()
        ax2.set_ylabel('Length in degree')
        ax3.grid()
        ax3.legend()
        ax3.set_ylabel('Width in degree')
        #plt.savefig(save_dir+dataset+'.jpg')
        plt.show()

    else: # for gamma data
        intensity_selection = intensity.T.ravel() != 0
        length_selection = length.T.ravel() != 0
        width_selection = width.T.ravel() != 0
        energy_weight = np.repeat(header_energy,9)
        bins = 100
        # calculating histograms
        intensity_weighted_hist = plt.hist2d(impact.ravel()[intensity_selection], np.log10(intensity.T.ravel()[intensity_selection]), bins=bins, weights=energy_weight[intensity_selection])
        intensity_unweighted_hist = plt.hist2d(impact.ravel()[intensity_selection], np.log10(intensity.T.ravel()[intensity_selection]), bins=bins)
        length_weighted_hist = plt.hist2d(impact.ravel()[length_selection], np.log10(intensity.T.ravel()[length_selection]), bins=bins, weights=length.T.ravel()[length_selection])
        length_unweighted_hist = plt.hist2d(impact.ravel()[length_selection], np.log10(intensity.T.ravel()[length_selection]), bins=bins)
        length_squared_hist = plt.hist2d(impact.ravel()[length_selection], np.log10(intensity.T.ravel()[length_selection]),bins=bins, weights=(length.T.ravel()[length_selection]) ** 2)
        width_weighted_hist = plt.hist2d(impact.ravel()[width_selection], np.log10(intensity.T.ravel()[width_selection]), bins=bins, weights=width.T.ravel()[width_selection])
        width_unweighted_hist = plt.hist2d(impact.ravel()[width_selection], np.log10(intensity.T.ravel()[width_selection]), bins=bins) #unweighted
        width_squared_hist = plt.hist2d(impact.ravel()[width_selection], np.log10(intensity.T.ravel()[width_selection]), bins=bins, weights=(width.T.ravel()[width_selection])**2)

        intensity_division = np.divide(intensity_weighted_hist[0],intensity_unweighted_hist[0])
        length_division = np.divide(length_weighted_hist[0],length_unweighted_hist[0])
        length_squared_division = np.divide(length_squared_hist[0], length_unweighted_hist[0])
        width_division = np.divide(width_weighted_hist[0],width_unweighted_hist[0])
        width_squared_division = np.divide(width_squared_hist[0],width_unweighted_hist[0])

        # Approximation of std's for histograms
        stdLengthapprox = np.zeros(np.shape(length_division.ravel()))
        stdWidthapprox = np.zeros(np.shape(width_division.ravel()))
        for i in range(0, len(length_division.ravel())):
            stdLengthapprox[i] = (length_squared_division.ravel()[i] - (length_division.ravel()[i]) ** 2)
            stdWidthapprox[i]  = (width_squared_division.ravel()[i]  - (width_division.ravel()[i]) ** 2)
            i += 1
        stdWidthapprox = np.sqrt(np.reshape(stdWidthapprox, np.shape(width_division)))
        stdLengthapprox = np.sqrt(np.reshape(stdLengthapprox, np.shape(length_division)))
        # at this point stdLengthapprox and stdWidthapprox are made into std instead of variance

        # plotting of histograms to doublecheck
    def make_hist_plot(name):
        fig, ax = plt.subplots(figsize=(6,5))
        if name=='intensity' or name =='Intensity' or name=='energy' or name=='Energy':
            fig.suptitle('Lookup table for expected Energy')
            im = ax.imshow(intensity_division, origin='lower', extent=(0,np.max(intensity_weighted_hist[1]),0,np.max(intensity_weighted_hist[2])), aspect='auto')
            ax.set_xlabel('Impact Distance in meters')
            ax.set_ylabel('log10(Intensity) in PE')
            ax.set_xlim((0, 400))
            ax.set_ylim((0, 4))
            ax.set_xticks(np.linspace(0,400,9))
            ax.set_yticks(np.linspace(0,4,9))
            plt.colorbar(im, label='Avg Energy in PE')
        elif name=='length' or name=='Length':
            fig.suptitle('Lookup table for Avg Length in degrees')
            im = ax.imshow(length_division, origin='lower', extent=(0,np.max(length_weighted_hist[1]),0,np.max(length_weighted_hist[2])), aspect='auto')
            ax.set_xlabel('Impact Distance in meters')
            ax.set_ylabel('log10(Intensity) in PE')
            ax.set_xlim((0, 400))
            ax.set_ylim((0, 4))
            ax.set_xticks(np.linspace(0, 400, 9))
            ax.set_yticks(np.linspace(0, 4, 9))
            plt.colorbar(im, label='Avg Length in degrees')
        elif name=='width'or name=='Width':
            fig.suptitle('Lookup table for Avg Width in degrees')
            im = ax.imshow(width_division, origin='lower', extent=(0,np.max(width_weighted_hist[1]),0,np.max(width_weighted_hist[2])), aspect='auto')
            ax.set_xlabel('Impact Distance in meters')
            ax.set_ylabel('log10(Intensity) in PE')
            ax.set_xlim((0, 400))
            ax.set_ylim((0, 4))
            ax.set_xticks(np.linspace(0, 400, 9))
            ax.set_yticks(np.linspace(0, 4, 9))
            plt.colorbar(im, label='Avg Width in degrees')
        elif name=='std length'or name=='std Length':
            fig.suptitle('Lookup table for Std Length in degrees')
            im = ax.imshow(stdLengthapprox, origin='lower', extent=(0,np.max(length_weighted_hist[1]),0,np.max(length_weighted_hist[2])), aspect='auto')
            ax.set_xlabel('Impact Distance in meters')
            ax.set_ylabel('log10(Intensity) in PE')
            ax.set_xlim((0, 400))
            ax.set_ylim((0, 4))
            ax.set_xticks(np.linspace(0, 400, 9))
            ax.set_yticks(np.linspace(0, 4, 9))
            plt.colorbar(im, label='Std Length in degrees')
        elif name=='std width'or name=='std Width':
            fig.suptitle('Lookup table for Std Width in degrees')
            im = ax.imshow(stdWidthapprox, origin='lower', extent=(0,np.max(width_weighted_hist[1]),0,np.max(width_weighted_hist[2])), aspect='auto')
            ax.set_xlabel('Impact Distance in meters')
            ax.set_ylabel('log10(Intensity) in PE')
            ax.set_xlim((0, 400))
            ax.set_ylim((0, 4))
            ax.set_xticks(np.linspace(0, 400, 9))
            ax.set_yticks(np.linspace(0, 4, 9))
            plt.colorbar(im, label='Std Width in degrees')
        plt.show()
        #plt.savefig(save_dir+dataset+'.jpg')

    # this part is just for 'bug searching' via binning of the energy ranges
        zerototen = []
        tentoonehundred = []
        onehundredtoonethousand = []
        onethousandtotenthousand = []
        tenthousandtoonehundredthousand = []
        for index in range(0, int(len(header['energy']) / 10)):
            if header['energy'][index] < 10:
                zerototen = np.append(zerototen, [impact.ravel()[i] for i in range(index, index + 10)])
            elif header['energy'][index] >= 10 and header['energy'][index] < 100:
                tentoonehundred = np.append(tentoonehundred, [impact.ravel()[i] for i in range(index * 10, (index * 10) + 10)])
            elif header['energy'][index] >= 100 and header['energy'][index] < 1000:
                onehundredtoonethousand = np.append(onehundredtoonethousand, [impact.ravel()[i] for i in range(index * 10, (index * 10) + 10)])
            elif header['energy'][index] >= 1000 and header['energy'][index] < 10000:
                onethousandtotenthousand = np.append(onethousandtotenthousand, [impact.ravel()[i] for i in range(index * 10, (index * 10) + 10)])
            elif header['energy'][index] >= 10000 and header['energy'][index] < 100000:
                tenthousandtoonehundredthousand = np.append(tenthousandtoonehundredthousand, [impact.ravel()[i] for i in range(index * 10, (index * 10) + 10)])
            else:
                print('failed at', index)

    print('Now writing:'+ dataset+".txt")
    # writing everything into a textfile for later use
    textfile = open(save_dir+dataset+".txt", "w+")
    #textfile.write("#average_intensity \n")
    #np.savetxt(textfile, avg_intensity)
    #textfile.write("#average_length \n")
    #np.savetxt(textfile, avg_length)
    #textfile.write("#average_width \n")
    #np.savetxt(textfile, avg_width)
    #textfile.write("#std_intensity \n")
    #np.savetxt(textfile, std_intensity)
    #textfile.write("#std_length \n")
    #np.savetxt(textfile, std_length)
    #textfile.write("#std_width \n")
    #np.savetxt(textfile, std_width)
    #textfile.write('#tenth_percentile_intensity \n')
    #np.savetxt(textfile, percentile_intensity)
    #textfile.write('#tenth_percentile_length \n')
    #np.savetxt(textfile, percentile_length)
    #textfile.write('#tenth_percentile_width \n')
    #np.savetxt(textfile, percentile_width)
    textfile.write('#intensity \n')
    np.savetxt(textfile, intensity.T.ravel())
    textfile.write('#length \n')
    np.savetxt(textfile, length.T.ravel())
    textfile.write('#width \n')
    np.savetxt(textfile, width.T.ravel())
    textfile.write('#energy \n')
    np.savetxt(textfile, header_energy)
    if "proton" in dataset:
        textfile.close()
    else:
        textfile.write('#impact_dist \n')
        np.savetxt(textfile, impact.ravel())
        #for intensity plots
        textfile.write('#intensity_hist \n')
        np.savetxt(textfile, intensity_division.ravel())
        textfile.write('#intensity_hist_xmax \n')
        np.savetxt(textfile, intensity_weighted_hist[1])
        textfile.write('#intensity_hist_ymax \n')
        np.savetxt(textfile, intensity_weighted_hist[2])
        #for length plots
        textfile.write('#length_hist \n')
        np.savetxt(textfile, length_division.ravel())
        textfile.write('#length_hist_xmax \n')
        np.savetxt(textfile, length_weighted_hist[1])
        textfile.write('#length_hist_ymax \n')
        np.savetxt(textfile, length_weighted_hist[2])
        #for width plots
        textfile.write('#width_hist \n')
        np.savetxt(textfile, width_division.ravel())
        textfile.write('#width_hist_xmax \n')
        np.savetxt(textfile, width_weighted_hist[1])
        textfile.write('#width_hist_ymax \n')
        np.savetxt(textfile, width_weighted_hist[2])
        #for Std Plots
        textfile.write('#stdLengthapprox \n')
        np.savetxt(textfile, stdLengthapprox.ravel())
        textfile.write('#stdWidthapprox \n')
        np.savetxt(textfile, stdWidthapprox.ravel())
        textfile.close()