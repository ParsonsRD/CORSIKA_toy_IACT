import numpy as np                                              # importing usefull packages
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import pkg_resources
from scipy.stats import norm
#from scipy.ndimage import gaussian_filter


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

    # Interpolation used later
def find_exp_value(lookuptable,xaxis ,yaxis ,xval,yval):
    xaxis_cent = xaxis[:-1] + np.diff(xaxis)/2
    yaxis_cent = yaxis[:-1] + np.diff(yaxis)/2
    interpol = RegularGridInterpolator((xaxis_cent, yaxis_cent), lookuptable.T, method='nearest' ,bounds_error=False, fill_value=-100) #method='nearest' gives the choppy pixel-by-pixel value instead of this "gradient" rn
    exp_val = interpol((xval,yval))
    return exp_val #print('exp val using interpolation:', exp_val),

    #
def calc_SC_Value(val, exp, std, boundary = 10000):
    '''calculates Scaled Value, Width or Length
        Inputs:
            val: array containing actual values '''
    SC_val = (val - exp) / std
    SC_nonnan = SC_val[~np.isnan(SC_val)]
    SC_noninf = SC_nonnan[~np.isinf(SC_nonnan)]
    SC_no_min = SC_noninf[SC_noninf>-boundary]
    SC_no_max = SC_no_min[SC_no_min<boundary]
    return SC_no_max

def add_to_array(a,b):
    '''Input: a, b Arrays of different size that need to be added up
       Output: c Array consisting of a+b with size of the larger one'''
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c

def gaussian_filter(raw_image, sigma=1, num_sigmas=5):
    shape = raw_image.shape
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    output_image = np.zeros_like(raw_image)

    for i in range(shape[0]):
        for j in range(shape[1]):
            r = np.sqrt((x - j) ** 2 + (y - i) ** 2)
            prob = norm.pdf(r, scale=sigma)

            sel = np.logical_and(r < num_sigmas * sigma, ~np.isnan(raw_image))
            # sel = np.logical_and(sel, raw_image>0)

            value = np.sum(raw_image[sel] * prob[sel])
            weight = np.sum(prob[sel])

            output_image[i][j] = value / weight

    return output_image



    # setting up the filepaths
save_dir = pkg_resources.resource_filename('corsika_toy_iact', 'save_dir/gammas_0deg/')
#gamma001 = save_dir+"./CER000101.root.txt"

SCL_array = []
SCW_array = []

for i in range (1,6): #201
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

    # actually reading in and naming the dataset
    filepath_i = save_dir+data_i
    res = readfile(filepath_i)
    bins = 100
    intensity   = np.reshape(np.log10(res['#intensity']), (10000, 9))
    length      = np.reshape(res['#length'], (10000, 9))
    width       = np.reshape(res['#width'], (10000, 9))
    energy = res['#energy']
    impact = np.reshape(res['#impact_dist'], (10000, 9))

    gaus_sigma = 4
    #energy_hist = gaussian_filter(np.reshape(res['#intensity_hist'], (bins,bins)), sigma = gaus_sigma)
    energy_hist = np.reshape(res['#intensity_hist'], (bins,bins))
    #energy_hist[np.isnan(energy_hist)] = 0
    energy_hist = gaussian_filter(energy_hist, sigma = gaus_sigma)
    energy_hist_x = res['#intensity_hist_xmax']
    energy_hist_y = res['#intensity_hist_ymax']

    #length_hist = gaussian_filter(np.reshape(res['#length_hist'], (bins,bins)), sigma = gaus_sigma)
    length_hist = np.reshape(res['#length_hist'], (bins,bins))
    #length_hist[np.isnan(length_hist)] = 0
    length_hist = gaussian_filter(length_hist, sigma= gaus_sigma)
    length_hist_x = res['#length_hist_xmax']
    length_hist_y = res['#length_hist_ymax']

    #width_hist = gaussian_filter(np.reshape(res['#width_hist'], (bins,bins)), sigma = gaus_sigma)
    width_hist = np.reshape(res['#width_hist'], (bins,bins))
    #width_hist[np.isnan(width_hist)] = 0
    width_hist = gaussian_filter(width_hist, sigma = gaus_sigma)
    width_hist_x = res['#width_hist_xmax']
    width_hist_y = res['#width_hist_ymax']

    #stdLengthapprox = gaussian_filter(np.reshape(res['#stdLengthapprox'], (bins, bins)), sigma = gaus_sigma)
    stdLengthapprox = np.reshape(res['#stdLengthapprox'], (bins, bins))
    #stdLengthapprox[np.isnan(stdLengthapprox)] = 0
    stdLengthapprox = gaussian_filter(stdLengthapprox, sigma=gaus_sigma)
    #stdWidthapprox = gaussian_filter(np.reshape(res['#stdWidthapprox'], (bins,bins)), sigma = gaus_sigma)
    stdWidthapprox = np.reshape(res['#stdWidthapprox'], (bins,bins))
    #stdWidthapprox[np.isnan(stdWidthapprox)] = 0
    stdWidthapprox = gaussian_filter(stdWidthapprox, sigma=gaus_sigma)

    # If the txt contains these, idk if it will in the final txtsave, maybe useful lateron
    #    bins = res['bins']

        # data manipulation to include only NON-ZERO events
    length_selection = length.ravel() != 0
    width_selection = width.ravel() != 0

    test_length_value = length.ravel()[length_selection]
    test_length_impact = impact.ravel()[length_selection]
    test_length_intensity = intensity.ravel()[length_selection]

    test_width_value = width.ravel()[width_selection]
    test_width_impact = impact.ravel()[width_selection]
    test_width_intensity = intensity.ravel()[width_selection]

    # interpolation
    exp_length_interpol = find_exp_value(length_hist, length_hist_x, length_hist_y, test_length_impact, test_length_intensity)
    exp_std_length_interpol = find_exp_value(stdLengthapprox, length_hist_x, length_hist_y, test_length_impact, test_length_intensity)

    exp_width_interpol = find_exp_value(width_hist, width_hist_x, width_hist_y, test_width_impact, test_width_intensity)
    exp_std_width_interpol = find_exp_value(stdWidthapprox, width_hist_x, width_hist_y, test_width_impact, test_width_intensity)

    #calculating Scaled Values
    SCL = calc_SC_Value(test_length_value, exp_length_interpol, exp_std_length_interpol, 1000000000)
    SCW = calc_SC_Value(test_width_value, exp_width_interpol, exp_std_width_interpol, 1000000000)

    SCL_array.append(np.average(SCL))
    SCW_array.append(np.average(SCW))

    # histogram distribution plots of SCL and SCW ## CURRENTLY NOT WORKING AS INTENDED
    if i == 1:
        SCL_hist_array = np.zeros(SCL.shape)
        SCW_hist_array = np.zeros(SCW.shape)
        SCL_hist_array = np.add(SCL_hist_array, SCL)
        SCW_hist_array = np.add(SCW_hist_array, SCW)
    else:
        SCL_hist_array = add_to_array(SCL_hist_array, SCL)
        SCW_hist_array = add_to_array(SCW_hist_array, SCW)



    #plt.hist(SCW_noninf, bins=100, range=(-10,10)) # useful for seeing the distribution of the SCW to determine if gaussian
    plt.hist(SCL, bins=100, range=(-10,10)) # useful for seeing the distribution of the SCW to determine if gaussian
    plt.title('SCL Distribution')

# doing a testplot of the 2D- histogram to see if everything imported fine/to inspect the lookup table

#fig, ax = plt.subplots(figsize=(6,5))
#fig.suptitle('Lookup table for expected Energy')
    #im = ax.imshow(energy_hist, origin='lower', extent=(np.min(energy_hist_x),np.max(energy_hist_x),np.min(energy_hist_y),np.max(energy_hist_y)), aspect='auto')
#im = ax.imshow(stdLengthapprox, origin='lower', extent=(np.min(length_hist_x),np.max(length_hist_x),np.min(length_hist_y),np.max(length_hist_y)), aspect='auto')
#ax.set_xlabel('Impact Distance in meters')
#ax.set_ylabel('log10(Intensity) in PE')
#ax.set_xlim((0, 400))
#ax.set_ylim((1.5, 4))
#ax.set_xticks(np.linspace(0,400,9))
#ax.set_yticks(np.linspace(1.5,4,6))
    #plt.colorbar(im, label='Avg Energy in PE')
#plt.colorbar(im, label='std of Length')
#plt.show()



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_index(array, value):
    index = array.tolist().index(value)
    return index

def find_expectation_value(lookuptable,xaxis ,yaxis ,xval,yval): #first version
    interpolating = RegularGridInterpolator((np.linspace(1, bins, bins), np.linspace(1, bins, bins)), lookuptable.T)
    xindex = find_index(xaxis, find_nearest(xaxis, xval))
    yindex = find_index(yaxis, find_nearest(yaxis, yval))
    exp_val2 =  lookuptable.T[xindex][yindex] #now gives a z value like the colorbar for that specific pixel/bin, but interpolation counts different
    expectation_val = interpolating((xindex+1,yindex+1))
    return print('exp val using hist:', exp_val2, '\nexp val using interpolation:',expectation_val, '\n', 'x:', xindex, 'y:', yindex)

SCL_val = (test_length_value - exp_length_interpol ) / exp_std_length_interpol
SCL_nonnan = SCL_val[~np.isnan(SCL_val)]
SCL_noninf = SCL_nonnan[~np.isinf(SCL_nonnan)]
SCL_min = SCL_noninf[SCL_noninf<-10000]
#SCL_min.shape
test_length_value2 = test_length_value[~np.isnan(SCL_val)]
test_length_value3 = test_length_value2[~np.isinf(SCL_nonnan)]
test_length_value4 = test_length_value3[SCL_noninf<-10000]

exp_length_interpol2 = exp_length_interpol[~np.isnan(SCL_val)]
exp_length_interpol3 = exp_length_interpol2[~np.isinf(SCL_nonnan)]
exp_length_interpol4 = exp_length_interpol3[SCL_noninf<-10000]

exp_std_length_interpol2 = exp_std_length_interpol[~np.isnan(SCL_val)]
exp_std_length_interpol3 = exp_std_length_interpol2[~np.isinf(SCL_nonnan)]
exp_std_length_interpol4 = exp_std_length_interpol3[SCL_noninf<-10000]
#np.min(SCL)
#min_index = find_index(SCL, find_nearest(SCL, np.min(SCL)))
#SCL[min_index]
#(test_length_value3[min_index] - exp_length_interpol3[min_index]) / exp_std_length_interpol3[min_index]
