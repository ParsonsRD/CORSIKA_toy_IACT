import numpy as np
import math
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import pkg_resources

save_dir = pkg_resources.resource_filename('corsika_toy_iact', 'save_dir/InterpolTest/')
gamma001 = save_dir+"./CER000101.root.txt"

def readfile(datafile):
    '''Input: .txt datafile you want to read in OR name of the variable linked to the right .txt
        Output: dictionary where 0th component is avg_intensity, 1st ist avg_length, 2nd is avg_width, 3rd is tenth_percentile_intensity and so on'''
    bigarray = np.genfromtxt(datafile, comments=None,dtype=str)
    ind = np.arange(0, bigarray.size, 1)
    res = {}
    marker = ind[np.char.startswith(bigarray, '#')]
    for i in range(len(marker)):

        if i == len(marker) - 1:
            res[bigarray[marker[i]]] = np.asarray(bigarray[marker[i] + 1:], dtype=float)
            break
        res[bigarray[marker[i]]] = np.asarray(bigarray[marker[i] + 1: marker[i + 1]], dtype=float)

    return res


res = readfile(gamma001)
bins = 100
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

# If the txt contains these, idk if it will in the final txtsave
#    avg_intensity   = res['#average_intensity']
#    avg_length      = res['#average_length']
#    avg_width       = res['#average_width']
#    std_intensity   = res['#std_intensity']
#    std_length      = res['#std_length']
#    std_width       = res['#std_width']
intensity   = np.reshape(np.log10(res['#intensity']), (10000, 9))
length      = np.reshape(res['#length'], (10000, 9))
width       = np.reshape(res['#width'], (10000, 9))
#    bins = res['bins']

#testplot to see if everything imported fine
fig, ax = plt.subplots(figsize=(6,5))
fig.suptitle('Lookup table for expected Energy')
im = ax.imshow(energy_hist, origin='lower', extent=(np.min(energy_hist_x),np.max(energy_hist_x),np.min(energy_hist_y),np.max(energy_hist_y)), aspect='auto')
#im = ax.imshow(intensity_hist, origin='lower', aspect='auto')
ax.set_xlabel('Impact Distance in meters')
ax.set_ylabel('log10(Intensity) in PE')
ax.set_xlim((0, 400))
ax.set_ylim((1.5, 4))
ax.set_xticks(np.linspace(0,400,9))
ax.set_yticks(np.linspace(1.5,4,6))
plt.colorbar(im, label='Avg Energy in PE')
plt.show()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_index(array, value):
    index = array.tolist().index(value)
    return index

def find_expectation_value(lookuptable,xaxis ,yaxis ,xval,yval):
    interpolating = RegularGridInterpolator((np.linspace(1, bins, bins), np.linspace(1, bins, bins)), lookuptable.T)
    xindex = find_index(xaxis, find_nearest(xaxis, xval))
    yindex = find_index(yaxis, find_nearest(yaxis, yval))
    exp_val2 =  lookuptable.T[xindex][yindex] #now gives a z value like the colorbar for that specific pixel/bin, but interpolation counts different
    expectation_val = interpolating((xindex+1,yindex+1))
    return print('exp val using hist:', exp_val2, '\nexp val using interpolation:',expectation_val, '\n', 'x:', xindex, 'y:', yindex)

def find_exp_value(lookuptable,xaxis ,yaxis ,xval,yval):
    xaxis_cent = xaxis[:-1] + np.diff(xaxis)/2
    yaxis_cent = yaxis[:-1] + np.diff(yaxis)/2
    interpol = RegularGridInterpolator((xaxis_cent, yaxis_cent), lookuptable.T, method='nearest' ,bounds_error=False, fill_value=-100) #method='nearest' gives the choppy pixel-by-pixel value instead of this "gradient" rn
    exp_val = interpol((xval,yval))
    return exp_val #print('exp val using interpolation:', exp_val),

length_selection = length.ravel() != 0
width_selection = width.ravel() != 0

test_width_value = width.ravel()[width_selection]
test_width_impact = impact.ravel()[width_selection]
test_width_intensity = intensity.ravel()[width_selection]

exp_width_interpol = find_exp_value(width_hist, width_hist_x, width_hist_y, test_width_impact, test_width_intensity)
exp_std_width_interpol = find_exp_value(stdWidthapprox, width_hist_x, width_hist_y, test_width_impact, test_width_intensity)

SCW = (test_width_value - exp_width_interpol) / exp_std_width_interpol
SCW_nonnan = SCW[~np.isnan(SCW)]
SCW_noninf = SCW_nonnan[~np.isinf(SCW_nonnan)]

plt.hist(SCW_noninf, bins=100, range=(-100,100))

#check hist without defiding by standard dev of SCW, check if its centered on 0,
#