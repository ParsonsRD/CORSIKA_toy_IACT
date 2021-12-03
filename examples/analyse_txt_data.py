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
    interpol = RegularGridInterpolator((xaxis_cent, yaxis_cent), lookuptable, method='nearest' ,bounds_error=False, fill_value=-100) #method='nearest' gives the choppy pixel-by-pixel value instead of this "gradient" rn
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


#starting the real code:
## READING IN FILES
starttime = time.time()
print('Starting to run and read in the Gamma file')
save_dir = pkg_resources.resource_filename('corsika_toy_iact', 'save_dir/')
filepath = save_dir+'Gamma_run1.txt' #Gamma_run_0_first50images
Gamma_res = readfile(filepath)
print('file read in')

print('Now reading in the proton file')
filepath2 = save_dir+'EPOS_test2.txt' #Gamma_run_0_first50images
Proton_res = readfile(filepath2)
print('file read in')

bins = (np.linspace(-5,805,82), np.linspace(-0.05,7.55,77))
        # made this way that the center of the leftmost bin is 0 increasing in even steps to cover the whole (impact, log10(intensity)) range

## INITIALIZING ARRAYS NEEDED
intensity = np.reshape(Gamma_res['#intensity'], (int(Gamma_res['#intensity'].shape[0]/9), 9))
length    = np.reshape(Gamma_res['#length'], (int(Gamma_res['#length'].shape[0]/9), 9))
width     = np.reshape(Gamma_res['#width'], (int(Gamma_res['#width'].shape[0]/9), 9))               # these are all shape (2000070, 9)
energy    = Gamma_res['#energy']                                                              # in GeV , also shape (2000070,)
impact    = np.reshape(Gamma_res['#impact_dist'], (int(Gamma_res['#impact_dist'].shape[0]/9), 9))
energy_weight = np.repeat(energy, 9)
intensity_selection = intensity != 0
length_selection = length != 0
width_selection = width != 0

p_intensity = np.reshape(Proton_res['#intensity'], (int(Proton_res['#intensity'].shape[0]/11), 11))
p_length    = np.reshape(Proton_res['#length'], (int(Proton_res['#length'].shape[0]/11), 11))
p_width     = np.reshape(Proton_res['#width'], (int(Proton_res['#width'].shape[0]/11), 11))
p_energy    = Proton_res['#energy']
p_position  = Proton_res['#positions'] #np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400])  UNTIL I FIX THE TXT FILE
p_energy_weight = np.repeat(p_energy, 11)

## MAKING HISTS
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

## SMOOTHING HISTS -> LOOKUP TABLES
gaussigma=2
width_hist = gaussian_filter(width_div, sigma=gaussigma)
length_hist = gaussian_filter(length_div, sigma=gaussigma)
stdWidthapprox = gaussian_filter(stdWidthapprox, sigma=gaussigma)
stdLengthapprox = gaussian_filter(stdLengthapprox, sigma=gaussigma)

## TO SHOW HISTS
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

## GETTING GAMMA VALUES FOR FURTHER CALCULATIONS
intensity_selection = intensity != 0
length_selection = length != 0
width_selection = width != 0

length_value = length[length_selection]
length_impact = impact[length_selection]
length_intensity = intensity[length_selection]
width_value = width[width_selection]
width_impact = impact[width_selection]
width_intensity = intensity[width_selection]

## IMPORTANT ##
#Something something (xval,yval) for interpolation into the right shape like follows

# p_position has shape (11,) which i can use as my x coord, in steps of 40m
    # p_intensity has shape (100000, 11) meaning that for all of the 11 telescopes, there are 100k events
# p_intensity.T[0] gives me all events for the 0th telescope (at x=0) and so on
# so i need to get it into the right ZUSAMMENSCHLUSS of x and y for the interpolation
# kind of like p_interpol((p_position,p_intensity.T)) but using only the 1st dimension of the intensity array and disregarding the 2nd which's (11,)


## GETTING PROTON VALUES FOR FURTHER CALCULATIONS
p_intensity_selection = []
p_length_selection = []
p_width_selection = []
final_p_pos = []

final_p_intens = []
final_p_length = []
final_p_width = []

for i in range (0, len(p_position)):
    p_intensity_selection_i = p_intensity.T[i] != 0
    p_length_selection_i = p_length.T[i] != 0
    p_width_selection_i = p_width.T[i] != 0
    if i == 0:
        p_intensity_selection = p_intensity_selection_i
        p_length_selection = p_length_selection_i
        p_width_selection = p_width_selection_i
    else:
        p_intensity_selection = np.concatenate((p_intensity_selection, p_intensity_selection_i)) #since these are masks we need to use concatenate and not append
        p_length_selection = np.concatenate((p_length_selection, p_length_selection_i))
        p_width_selection = np.concatenate((p_width_selection, p_width_selection_i))

    i_pos = np.repeat(p_position[i], p_intensity.T[i].size)
    final_p_pos = np.append(final_p_pos, i_pos)
    final_p_intens = np.append(final_p_intens, p_intensity.T[i])
    final_p_length = np.append(final_p_length, p_length.T[i])
    final_p_width = np.append(final_p_width, p_width.T[i])


p_exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], final_p_pos[p_length_selection], np.log10(final_p_intens[p_length_selection]))
p_exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2], final_p_pos[p_length_selection], np.log10(final_p_intens[p_length_selection]))
p_exp_length_interpol2 = find_exp_value(length_hist, length_w[1], length_w[2], final_p_pos, np.log10(final_p_intens))
p_exp_std_length_interpol2 = find_exp_value(stdLengthapprox, length_w[1], length_w[2], final_p_pos, np.log10(final_p_intens))

p_exp_width_interpol = find_exp_value(width_hist, width_w[1], width_w[2], final_p_pos[p_width_selection], np.log10(final_p_intens[p_width_selection]))
p_exp_std_width_interpol = find_exp_value(stdWidthapprox, width_w[1], width_w[2], final_p_pos[p_width_selection], np.log10(final_p_intens[p_width_selection]))
p_exp_width_interpol2 = find_exp_value(width_hist, width_w[1], width_w[2], final_p_pos, np.log10(final_p_intens))
p_exp_std_width_interpol2 = find_exp_value(stdWidthapprox, width_w[1], width_w[2], final_p_pos, np.log10(final_p_intens))


p_SCL = (final_p_length[p_length_selection] - p_exp_length_interpol) / p_exp_std_length_interpol
p_SCL2 = (final_p_length[p_length_selection] - p_exp_length_interpol2[p_length_selection]) / p_exp_std_length_interpol2[p_length_selection]

p_SCW = (final_p_width[p_width_selection] - p_exp_width_interpol) / p_exp_std_width_interpol
p_SCLW = (final_p_width[p_width_selection] - p_exp_width_interpol2[p_width_selection]) / p_exp_std_width_interpol2[p_width_selection]

bin_energies = False
if bin_energies == True: # Binning energy (in GeV) to create selection
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

    plt.clf()
    plt.hist(Zeroto100GeV_SCL.ravel(), bins=100, range=(-2.5,2.5), color='red', label='0GeV to 100GeV')
    plt.hist(HundredGeVto1TeV_SCL.ravel(), bins=100, range=(-2.5,2.5), color='blue', label='1000GeV to 1TeV')
    plt.hist(OneTeVto10TeV_SCL.ravel(), bins=100, range=(-2.5,2.5), color='green', label='1TeV to 10TeV')
    plt.hist(TenTeVto100TeV_SCL.ravel(), bins=100, range=(-2.5,2.5), color='yellow', label='10TeV to 100TeV')
    plt.hist(MoreThan100TeV_SCL.ravel(), bins=100, range=(-2.5,2.5), color='black', label='> 100TeV')
    plt.legend()
    plt.show()
else:
    pass

gamma_exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], length_impact, np.log10(length_intensity))
gamma_exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2], length_impact, np.log10(length_intensity))

gamma_exp_width_interpol = find_exp_value(width_hist, width_w[1], width_w[2], width_impact, np.log10(width_intensity))
gamma_exp_std_width_interpol = find_exp_value(stdWidthapprox, width_w[1], width_w[2], width_impact, np.log10(width_intensity))

gamma_SCL = calc_SC_Value(length_value, gamma_exp_length_interpol, gamma_exp_std_length_interpol, 10000)
gamma_SCW = calc_SC_Value(width_value, gamma_exp_width_interpol, gamma_exp_std_width_interpol, 10000)


#print('avg SCL:',np.average(SCL),'std SCL:', np.std(SCL))
# useful for seeing the distribution of the SCW to determine if gaussian
#plt.hist(SCL, bins=100, range=(-5,5))
#plt.title('SCL Distribution')

endtime = time.time()
print('running took', endtime-starttime, 'seconds')

#print('Now writing:' + "Schmutz_test2.txt")
#textfile = open(save_dir + "Schmutz_test2.txt", "w+")
#
#textfile.close()
