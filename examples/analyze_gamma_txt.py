import numpy as np                                              # importing usefull packages
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import pkg_resources


# function for reading in txt
def readfile(datafile):
    '''Input: .txt datafile you want to read in OR name of the variable linked to the right .txt
        Output: dictionary where 0th component is avg_intensity, 1st ist avg_length, 2nd is avg_width, 3rd is tenth_percentile_intensity and so on'''
    bigarray = np.genfromtxt(datafile, comments=None,dtype=str)
    ind = np.arange(0, bigarray.size, 1)
    res = {}
    marker = ind[np.char.startswith(bigarray, '#')]
    for j in range(len(marker)):

        if j == len(marker) - 1:
            res[bigarray[marker[j]]] = np.asarray(bigarray[marker[j] + 1:], dtype=float)
            break
        res[bigarray[marker[j]]] = np.asarray(bigarray[marker[j] + 1: marker[j + 1]], dtype=float)

    return res

    # Actual interpolation used later
def find_exp_value(lookuptable,xaxis ,yaxis ,xval,yval):
    xaxis_cent = xaxis[:-1] + np.diff(xaxis)/2
    yaxis_cent = yaxis[:-1] + np.diff(yaxis)/2
    interpol = RegularGridInterpolator((xaxis_cent, yaxis_cent), lookuptable.T, method='nearest' ,bounds_error=False, fill_value=-100) #method='nearest' gives the choppy pixel-by-pixel value instead of this "gradient" rn
    exp_val = interpol((xval,yval))
    return exp_val #print('exp val using interpolation:', exp_val),

    # setting up the filepaths
save_dir = pkg_resources.resource_filename('corsika_toy_iact', 'save_dir/gammas_0deg/')
#gamma001 = save_dir+"./CER000101.root.txt"

length_value_array = []
length_interpol_array = []
std_length_interpol_array = []
SCL_array = []
avg_SCL_array = []

width_value_array = []
width_interpol_array = []
std_width_interpol_array = []
SCW_array = []
avg_SCW_array = []

for i in range (1,11): #201
    if i < 10:
        data_i = 'CER00000' + str(i) + '.image.npz.txt'
    elif i > 9 and i < 100 and i != 37 and i != 76 and i != 77 and i != 93:
        data_i = 'CER0000' + str(i) + '.image.npz.txt'
    elif i == 37 or i == 76 or i == 77 or i == 93 : continue
    elif i > 99 and i != 112 and i != 121 and i != 191:
        data_i = 'CER000' + str(i) + '.image.npz.txt'
    elif i == 112 or i == 121 or i == 191: continue
    elif i > 201 : break

    print('calculating dataset:', data_i)
#for i in range (10,100):
    #data_i ='CER0000' + str(i) + '.image.npz'
#for i in range (100,201):
    #data_i = 'CER000' + str(i) + '.image.npz'

    filepath_i = save_dir+data_i
    # actually reading in and naming the dataset
    res = readfile(filepath_i)
    bins = 100
    intensity   = np.reshape(np.log10(res['#intensity']), (10000, 9))
    length      = np.reshape(res['#length'], (10000, 9))
    width       = np.reshape(res['#width'], (10000, 9))
    energy = res['#energy']
    impact = np.reshape(res['#impact_dist'], (10000, 9))

    energy_hist = np.reshape(res['#intensity_hist'], (bins,bins))
    energy_hist_x = res['#intensity_hist_xmax']
    energy_hist_y = res['#intensity_hist_ymax']

    length_hist = np.reshape(res['#length_hist'], (bins,bins))
    length_hist_x = res['#length_hist_xmax']
    length_hist_y = res['#length_hist_ymax']

    width_hist = np.reshape(res['#width_hist'], (bins,bins))
    width_hist_x = res['#width_hist_xmax']
    width_hist_y = res['#width_hist_ymax']

    stdLengthapprox = np.reshape(res['#stdLengthapprox'], (bins, bins))
    stdWidthapprox = np.reshape(res['#stdWidthapprox'], (bins,bins))

    # If the txt contains these, idk if it will in the final txtsave, maybe useful lateron
    #    bins = res['bins']

    # funtions used in early version of extrapolation
#def find_nearest(array, value):
#    array = np.asarray(array)
#    idx = (np.abs(array - value)).argmin()
#        return array[idx]

#    def find_index(array, value):
#        index = array.tolist().index(value)
#        return index

#    def find_expectation_value(lookuptable,xaxis ,yaxis ,xval,yval): #first version
#        interpolating = RegularGridInterpolator((np.linspace(1, bins, bins), np.linspace(1, bins, bins)), lookuptable.T)
#        xindex = find_index(xaxis, find_nearest(xaxis, xval))
#        yindex = find_index(yaxis, find_nearest(yaxis, yval))
#        exp_val2 =  lookuptable.T[xindex][yindex] #now gives a z value like the colorbar for that specific pixel/bin, but interpolation counts different
#        expectation_val = interpolating((xindex+1,yindex+1))
#        return print('exp val using hist:', exp_val2, '\nexp val using interpolation:',expectation_val, '\n', 'x:', xindex, 'y:', yindex)



        # Testing on some data, trying to calculate SCALED WIDTH
    length_selection = length.ravel() != 0
    width_selection = width.ravel() != 0

    test_length_value = length.ravel()[length_selection]
    test_length_impact = impact.ravel()[length_selection]
    test_length_intensity = intensity.ravel()[length_selection]

    test_width_value = width.ravel()[width_selection]
    test_width_impact = impact.ravel()[width_selection]
    test_width_intensity = intensity.ravel()[width_selection]


    exp_length_interpol = find_exp_value(length_hist, length_hist_x, length_hist_y, test_length_impact, test_length_intensity)
    exp_std_length_interpol = find_exp_value(stdLengthapprox, length_hist_x, length_hist_y, test_length_impact, test_length_intensity)

    exp_width_interpol = find_exp_value(width_hist, width_hist_x, width_hist_y, test_width_impact, test_width_intensity)
    exp_std_width_interpol = find_exp_value(stdWidthapprox, width_hist_x, width_hist_y, test_width_impact, test_width_intensity)


    SCL = (test_length_value - exp_length_interpol) / exp_std_length_interpol
    SCL_nonnan = SCL[~np.isnan(SCL)]
    SCL_noninf = SCL_nonnan[~np.isinf(SCL_nonnan)]
    SCL_noninf2 = SCL_noninf[SCL_noninf>-10000]
    SCL_noninf3 = SCL_noninf2[SCL_noninf2<10000]
    avg_SCL  = (np.average(test_length_value[~np.isnan(test_length_value)]) - np.average(exp_length_interpol[~np.isnan(exp_length_interpol)])) / np.average(exp_std_length_interpol[~np.isnan(exp_std_length_interpol)])

    SCW = (test_width_value - exp_width_interpol) / exp_std_width_interpol
    SCW_nonnan = SCW[~np.isnan(SCW)]
    SCW_noninf = SCW_nonnan[~np.isinf(SCW_nonnan)]
    SCW_noninf2 = SCW_noninf[SCW_noninf>-10000]
    SCW_noninf3 = SCW_noninf2[SCW_noninf2<10000]
    avg_SCW = (np.average(test_width_value[~np.isnan(test_width_value)]) - np.average(exp_width_interpol[~np.isnan(exp_width_interpol)])) / np.average(exp_std_width_interpol[~np.isnan(exp_std_width_interpol)])

    length_value_array.append(np.average(test_length_value[~np.isnan(test_length_value)]))
    length_interpol_array.append(np.average(exp_length_interpol[~np.isnan(exp_length_interpol)]))
    std_length_interpol_array.append(np.average(exp_std_length_interpol[~np.isnan(exp_std_length_interpol)]))
    SCL_array.append(np.average(SCL_noninf3))
    avg_SCL_array.append(avg_SCL)


    width_value_array.append(np.average(test_width_value[~np.isnan(test_width_value)]))
    width_interpol_array.append(np.average(exp_width_interpol[~np.isnan(exp_width_interpol)]))
    std_width_interpol_array.append(np.average(exp_std_width_interpol[~np.isnan(exp_std_width_interpol)]))
    SCW_array.append(np.average(SCW_noninf3))
    avg_SCW_array.append(avg_SCW)

    #plt.hist(SCW_noninf, bins=100, range=(-10,10)) # useful for seeing the distribution of the SCW to determine if gaussian
    plt.hist(SCL_noninf3, bins=100, range=(-10,10)) # useful for seeing the distribution of the SCW to determine if gaussian
    plt.title('SCL Distribution')

# doing a testplot of the 2D- histogram to see if everything imported fine/to inspect the lookup table

fig, ax = plt.subplots(figsize=(6,5))
fig.suptitle('Lookup table for expected Energy')
im = ax.imshow(energy_hist, origin='lower', extent=(np.min(energy_hist_x),np.max(energy_hist_x),np.min(energy_hist_y),np.max(energy_hist_y)), aspect='auto')
ax.set_xlabel('Impact Distance in meters')
ax.set_ylabel('log10(Intensity) in PE')
ax.set_xlim((0, 400))
ax.set_ylim((1.5, 4))
ax.set_xticks(np.linspace(0,400,9))
ax.set_yticks(np.linspace(1.5,4,6))
plt.colorbar(im, label='Avg Energy in PE')
plt.show()