import numpy as np                                              # importing useful packages
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import pkg_resources
from scipy.stats import norm
import time
#from scipy.ndimage import gaussian_filter

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

starttime = time.time()
print('Starting to run and read in the file')
save_dir = pkg_resources.resource_filename('corsika_toy_iact', 'save_dir/')
filepath = save_dir+'Gamma_run1.txt' #Gamma_run_0_first50images
res = readfile(filepath)
print('file read in')
bins = 100

intensity = np.reshape(res['#intensity'], (int(res['#intensity'].shape[0]/9), 9))
length    = np.reshape(res['#length'], (int(res['#length'].shape[0]/9), 9))
width     = np.reshape(res['#width'], (int(res['#width'].shape[0]/9), 9))               # these are all shape (2000070, 9)
energy    = res['#energy']                                                              # in GeV , also shape (2000070,)
impact    = np.reshape(res['#impact_dist'], (int(res['#impact_dist'].shape[0]/9), 9))

intensity_selection = intensity != 0
length_selection = length != 0
width_selection = width != 0
energy_weight = np.repeat(energy, 9)

intensity_w = make_hist('intensity', bins=bins, weights=True,squared=False)
intensity_unw = make_hist('intensity', bins=bins, weights=False,squared=False)
length_w = make_hist('length', bins=bins, weights=True,squared=False)
length_unw = make_hist('length', bins=bins, weights=False,squared=False)
length_sq = make_hist('length', bins=bins, weights=True, squared=True)
width_w = make_hist('width', bins=bins, weights=True,squared=False)
width_unw = make_hist('width', bins=bins, weights=False,squared=False)
width_sq = make_hist('width', bins=bins, weights=True, squared=True)


intensity_div = np.divide(intensity_w[0], intensity_unw[0])
length_div = np.divide(length_w[0], length_unw[0])
length_sq_div = np.divide(length_sq[0],length_unw[0])
width_div = np.divide(width_w[0], width_unw[0])
width_sq_div = np.divide(width_sq[0],width_unw[0])

stdLengthapprox = np.zeros(np.shape(length_div.ravel()))
stdWidthapprox = np.zeros(np.shape(width_div.ravel()))
for i in range(0, len(length_div.ravel())):
    stdLengthapprox[i] = (length_sq_div.ravel()[i] - (length_div.ravel()[i]) ** 2)
    stdWidthapprox[i]  = (width_sq_div.ravel()[i]  - (width_div.ravel()[i]) ** 2)
    i += 1
stdLengthapprox = np.sqrt(np.reshape(stdLengthapprox, np.shape(length_div)))
stdWidthapprox = np.sqrt(np.reshape(stdWidthapprox, np.shape(width_div)))

gaussigma=2
width_hist = gaussian_filter(width_div, sigma=gaussigma)
length_hist = gaussian_filter(length_div, sigma=gaussigma)
stdWidthapprox = gaussian_filter(stdWidthapprox, sigma=gaussigma)
stdLengthapprox = gaussian_filter(stdLengthapprox, sigma=gaussigma)

#fig, ax = plt.subplots(figsize=(6, 5))
#fig.suptitle('Lookup table for expected length')
#im = ax.imshow(length_hist, origin='lower',extent=(0, np.max(length_w[1]), 0, np.max(length_w[2])), aspect='auto')
#ax.set_xlabel('Impact Distance in meters')
#ax.set_ylabel('log10(Intensity) in PE')
#ax.set_xlim((0, 400))
#ax.set_ylim((0, 4))
#ax.set_xticks(np.linspace(0, 400, 9))
#ax.set_yticks(np.linspace(0, 4, 9))
#plt.colorbar(im, label='Avg Length in degree')

test_length_value = length[length_selection]
test_length_impact = impact[length_selection]
test_length_intensity = intensity[length_selection]


# Binning energy (in GeV) to create selection
Zeroto100GeV = np.logical_and(energy > 0 , energy <= 100)
HundredGeVto1TeV = np.logical_and(energy > 100 , energy <= 1000)
OneTeVto10TeV = np.logical_and(energy > 1000 , energy <= 10000)
TenTeVto100TeV = np.logical_and(energy > 10000 , energy <= 100000)
MoreThan100TeV = energy > 100000


# GeV Selection
Zeroto100GeV_length = length[Zeroto100GeV]
#Zeroto100GeV_length_selection = length_selection[Zeroto100GeV]
#Zeroto100GeV_length_value = Zeroto100GeV_length[Zeroto100GeV_length_selection]
Zeroto100GeV_impact = impact[Zeroto100GeV]
Zeroto100GeV_intensity = intensity[Zeroto100GeV]
Zeroto100GeV_exp_length_interpol =find_exp_value(length_hist, length_w[1], length_w[2], Zeroto100GeV_impact, np.log10(Zeroto100GeV_intensity))
Zeroto100GeV_std_length_interpol =find_exp_value(stdLengthapprox, length_w[1], length_w[2], Zeroto100GeV_impact, np.log10(Zeroto100GeV_intensity))
Zeroto100GeV_SCL = (Zeroto100GeV_length - Zeroto100GeV_exp_length_interpol) / Zeroto100GeV_std_length_interpol
# 100GeV - 1TeV
HundredGeVto1TeV_length = length[HundredGeVto1TeV]
#HundredGeVto1Tev_length_selection = length_selection[HundredGeVto1Tev] # why doesnt this have shape (2000070, 9) like Zeroto100GeV_length_selection does
#HundredGeVto1Tev_length_value = HundredGeVto1Tev_length[HundredGeVto1Tev_length_selection]
HundredGeVto1TeV_impact = impact[HundredGeVto1TeV]
HundredGeVto1TeV_intensity = intensity[HundredGeVto1TeV]
HundredGeVto1TeV_exp_length_interpol =find_exp_value(length_hist, length_w[1], length_w[2], HundredGeVto1TeV_impact, np.log10(HundredGeVto1TeV_intensity))
HundredGeVto1TeV_std_length_interpol =find_exp_value(stdLengthapprox, length_w[1], length_w[2], HundredGeVto1TeV_impact, np.log10(HundredGeVto1TeV_intensity))
HundredGeVto1TeV_SCL = (HundredGeVto1TeV_length - HundredGeVto1TeV_exp_length_interpol) / HundredGeVto1TeV_std_length_interpol
# 1-10 TeV
OneTeVto10TeV_length = length[OneTeVto10TeV]
#OneTeVto10TeV_length_selection = length_selection[OneTeVto10TeV]
#OneTeVto10TeV_length_value = OneTeVto10TeV_length[OneTeVto10TeV_length_selection]
OneTeVto10TeV_impact = impact[OneTeVto10TeV]
OneTeVto10TeV_intensity = intensity[OneTeVto10TeV]
OneTeVto10TeV_exp_length_interpol =find_exp_value(length_hist, length_w[1], length_w[2], OneTeVto10TeV_impact, np.log10(OneTeVto10TeV_intensity))
OneTeVto10TeV_std_length_interpol =find_exp_value(stdLengthapprox, length_w[1], length_w[2], OneTeVto10TeV_impact, np.log10(OneTeVto10TeV_intensity))
OneTeVto10TeV_SCL = (OneTeVto10TeV_length - OneTeVto10TeV_exp_length_interpol) / OneTeVto10TeV_std_length_interpol
# 10-100 TeV
TenTeVto100TeV_length = length[TenTeVto100TeV]
#TenTeVto100TeV_length_selection = length_selection[TenTeVto100TeV]
#TenTeVto100TeV_length_value = TenTeVto100TeV_length[TenTeVto100TeV_length_selection]
TenTeVto100TeV_impact = impact[TenTeVto100TeV]
TenTeVto100TeV_intensity = intensity[TenTeVto100TeV]
TenTeVto100TeV_exp_length_interpol =find_exp_value(length_hist, length_w[1], length_w[2], TenTeVto100TeV_impact, np.log10(TenTeVto100TeV_intensity))
TenTeVto100TeV_std_length_interpol =find_exp_value(stdLengthapprox, length_w[1], length_w[2], TenTeVto100TeV_impact, np.log10(TenTeVto100TeV_intensity))
TenTeVto100TeV_SCL = (TenTeVto100TeV_length - TenTeVto100TeV_exp_length_interpol) / TenTeVto100TeV_std_length_interpol
# MoreThan100TeV
MoreThan100TeV_length = length[MoreThan100TeV]
#MoreThan100TeV_length_selection = length_selection[MoreThan100TeV]
#MoreThan100TeV_length_value = MoreThan100TeV_length[MoreThan100TeV_length_selection]
MoreThan100TeV_impact = impact[MoreThan100TeV]
MoreThan100TeV_intensity = intensity[MoreThan100TeV]
MoreThan100TeV_exp_length_interpol =find_exp_value(length_hist, length_w[1], length_w[2], MoreThan100TeV_impact, np.log10(MoreThan100TeV_intensity))
MoreThan100TeV_std_length_interpol =find_exp_value(stdLengthapprox, length_w[1], length_w[2], MoreThan100TeV_impact, np.log10(MoreThan100TeV_intensity))
MoreThan100TeV_SCL = (MoreThan100TeV_length - MoreThan100TeV_exp_length_interpol) / MoreThan100TeV_std_length_interpol




test_width_value = width[width_selection]
test_width_impact = impact[width_selection]
test_width_intensity = intensity[width_selection]

exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], test_length_impact, np.log10(test_length_intensity))
exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2], test_length_impact, np.log10(test_length_intensity))

exp_width_interpol = find_exp_value(width_hist, width_w[1], width_w[2], test_width_impact, np.log10(test_width_intensity))
exp_std_width_interpol = find_exp_value(stdWidthapprox, width_w[1], width_w[2], test_width_impact, np.log10(test_width_intensity))

fill_select_w = exp_width_interpol == -100
fill_select_l = exp_length_interpol == -100

SCL = calc_SC_Value(test_length_value[~fill_select_l], exp_length_interpol[~fill_select_l], exp_std_length_interpol[~fill_select_l], 10000)
SCL2 = (test_length_value[~fill_select_l] - exp_length_interpol[~fill_select_l]) /  exp_std_length_interpol[~fill_select_l]
SCW = calc_SC_Value(test_width_value[~fill_select_w], exp_width_interpol[~fill_select_w], exp_std_width_interpol[~fill_select_w], 10000)
SCW2 = (test_width_value[~fill_select_w] - exp_width_interpol[~fill_select_w]) /  exp_std_width_interpol[~fill_select_w]
print('avg SCL:',np.average(SCL),'std SCL:', np.std(SCL))
# useful for seeing the distribution of the SCW to determine if gaussian
#plt.hist(SCL, bins=100, range=(-5,5))
#plt.title('SCL Distribution')

endtime = time.time()
print('running took', endtime-starttime, 'seconds')

plt.clf()
plt.hist(Zeroto100GeV_SCL.ravel(), bins=100, range=(-2.5,2.5), color='red', label='0GeV to 100GeV')
plt.hist(HundredGeVto1TeV_SCL.ravel(), bins=100, range=(-2.5,2.5), color='blue', label='1000GeV to 1TeV')
plt.hist(OneTeVto10TeV_SCL.ravel(), bins=100, range=(-2.5,2.5), color='green', label='1TeV to 10TeV')
plt.hist(TenTeVto100TeV_SCL.ravel(), bins=100, range=(-2.5,2.5), color='yellow', label='10TeV to 100TeV')
plt.hist(MoreThan100TeV_SCL.ravel(), bins=100, range=(-2.5,2.5), color='black', label='> 100TeV')
plt.legend()
plt.show()