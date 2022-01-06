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

def calc_tel_avg(array):
    avg_array = []
    for tel in range(array.T.shape[0]):
        avg_array = np.append(avg_array, np.average(array.T[tel]))
    return avg_array

def HP_comp_plot(epos,qgs,sibyll, param):
    plt.clf()
    plt.plot(p_position, epos / epos, label='EPOS', color='red')
    plt.plot(p_position, qgs / epos, label='QGS', color='green')
    plt.plot(p_position, sibyll / epos, label='SIBYLL', color='blue')
    plt.title('Avg {} HP at 1TeV compared to EPOS'.format(param))
    plt.xlabel('Telescope Position in m')
    plt.ylabel('relative {} Parameter'.format(param))
    plt.legend()
    plt.grid()
    plt.show

def passing_both(lengtharr,widtharr):
    '''ok so the theory here is that only those entries that are nonzero in the length and width array get used to
    calculate SCL/W down the line, so im basically deploying (very slow) loops that get the indices of those non-zero
    entries and save them in the 'SCX_indices' array,  then I loop over all indices that produce SCWs (because i know
    its smaller) and try to match each index in there with one from SCL and discard those that dont match,
    but apparently every entry in SCW matches with SCL but not other way around'''
    starttime_now=time.time()
    SCW_indices = []
    SCL_indices = []
    final_index_arr = []

    lengtharr_selection = lengtharr != 0
    widtharr_selection = widtharr != 0
    for i in range(widtharr_selection.size):
        if widtharr_selection.ravel()[i] == True:
            SCW_indices = np.append(SCW_indices, i)
        else:
            pass
    for j in range(lengtharr_selection.size):
        if lengtharr_selection.ravel()[j] == True:
            SCL_indices = np.append(SCL_indices, j)
        else:
            pass
    for k in range(SCW_indices.size):
        index_no = SCW_indices.ravel()[k]
        if index_no in SCL_indices:
            final_index_arr = np.append(final_index_arr, index_no)
        else:
            pass
    print('running took', time.time()-starttime_now, 'seconds')
    return final_index_arr

#starting the real code:
## READING IN FILES
starttime = time.time()
print('Starting to run and read in the Gamma file')
save_dir = pkg_resources.resource_filename('corsika_toy_iact', 'save_dir/')
filepath = save_dir+'Gamma_run1.txt' #Gamma_run_0_first50images
Gamma_res = readfile(filepath)
print('file read in')

print('Now reading in the first proton file: EPOS')
filename_p1= 'EPOS_1.txt'
filepath_p1 = save_dir+filename_p1 #Gamma_run_0_first50images
EPOS_res = readfile(filepath_p1)
print('file read in')

compareProton = True
if compareProton == True:
    print('Now reading in the 2nd proton file: QGS')
    filename_p2 = 'QGS_1.txt'
    filepath_p2 = save_dir + filename_p2  # Gamma_run_0_first50images
    QGS_res = readfile(filepath_p2)
    print('file read in')

    print('Now reading in the 3nd proton file: SIBYLL')
    filename_p3 = 'SIBYLL_1.txt'
    filepath_p3 = save_dir + filename_p3  # Gamma_run_0_first50images
    SIBYLL_res = readfile(filepath_p3)
    print('file read in')
else:
    pass

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

p_position  = EPOS_res['#positions'] #np.array([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400])  UNTIL I FIX THE TXT FILE

EPOS_intensity = np.reshape(EPOS_res['#intensity'], (int(EPOS_res['#intensity'].shape[0]/11), 11))
EPOS_length    = np.reshape(EPOS_res['#length'], (int(EPOS_res['#length'].shape[0]/11), 11))
EPOS_width     = np.reshape(EPOS_res['#width'], (int(EPOS_res['#width'].shape[0]/11), 11))
EPOS_energy    = EPOS_res['#energy']
EPOS_energy_weight = np.repeat(EPOS_energy, 11)

if compareProton == True:
    QGS_intensity = np.reshape(QGS_res['#intensity'], (int(QGS_res['#intensity'].shape[0] / 11), 11))
    QGS_length = np.reshape(QGS_res['#length'], (int(QGS_res['#length'].shape[0] / 11), 11))
    QGS_width = np.reshape(QGS_res['#width'], (int(QGS_res['#width'].shape[0] / 11), 11))
    QGS_energy = QGS_res['#energy']
    QGS_energy_weight = np.repeat(QGS_res, 11)

    SIBYLL_intensity = np.reshape(SIBYLL_res['#intensity'], (int(SIBYLL_res['#intensity'].shape[0] / 11), 11))
    SIBYLL_length = np.reshape(SIBYLL_res['#length'], (int(SIBYLL_res['#length'].shape[0] / 11), 11))
    SIBYLL_width = np.reshape(SIBYLL_res['#width'], (int(SIBYLL_res['#width'].shape[0] / 11), 11))
    SIBYLL_energy = SIBYLL_res['#energy']
    SIBYLL_energy_weight = np.repeat(SIBYLL_res, 11)
else:
    pass

## MAKING GAMMA HISTS
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
gaussigma=0.75
width_hist = gaussian_filter(width_div, sigma=gaussigma)
length_hist = gaussian_filter(length_div, sigma=gaussigma)
stdWidthapprox = gaussian_filter(stdWidthapprox, sigma=gaussigma)
stdLengthapprox = gaussian_filter(stdLengthapprox, sigma=gaussigma)

## TO SHOW HISTS
def showhist(hist):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle('Lookup table for expected length')
    im = ax.imshow(hist, origin='lower',extent=(0, np.max(length_w[1])-5, 0, np.max(length_w[2])-5), aspect='auto')
    ax.set_xlabel('Impact Distance in meters')
    ax.set_ylabel('log10(Intensity) in PE')
    ax.set_xlim((0, 400))
    ax.set_ylim((0, 4))
    ax.set_xticks(np.linspace(0, 400, 9))
    ax.set_yticks(np.linspace(0, 4, 9))
    plt.colorbar(im, label='Avg Length in degree')
    return plt.show()

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
EPOS_intensity_selection = []
EPOS_length_selection = []
EPOS_width_selection = []
final_EPOS_intens = []
final_EPOS_length = []
final_EPOS_width = []
final_EPOS_pos = []


for i in range (0, len(p_position)):
    EPOS_intensity_selection_i = EPOS_intensity.T[i] != 0
    EPOS_length_selection_i = EPOS_length.T[i] != 0
    EPOS_width_selection_i = EPOS_width.T[i] != 0
    if i == 0:
        EPOS_intensity_selection = EPOS_intensity_selection_i
        EPOS_length_selection = EPOS_length_selection_i
        EPOS_width_selection = EPOS_width_selection_i
    else:
        EPOS_intensity_selection = np.concatenate((EPOS_intensity_selection, EPOS_intensity_selection_i)) #since these are masks we need to use concatenate and not append
        EPOS_length_selection = np.concatenate((EPOS_length_selection, EPOS_length_selection_i))
        EPOS_width_selection = np.concatenate((EPOS_width_selection, EPOS_width_selection_i))

    i_pos = np.repeat(p_position[i], EPOS_intensity.T[i].size)
    final_EPOS_pos = np.append(final_EPOS_pos, i_pos)
    final_EPOS_intens = np.append(final_EPOS_intens, EPOS_intensity.T[i])
    final_EPOS_length = np.append(final_EPOS_length, EPOS_length.T[i])
    final_EPOS_width = np.append(final_EPOS_width, EPOS_width.T[i])

if compareProton == True:
    QGS_intensity_selection = []
    QGS_length_selection = []
    QGS_width_selection = []
    final_QGS_intens = []
    final_QGS_length = []
    final_QGS_width = []
    final_QGS_pos = []
    SIBYLL_intensity_selection = []
    SIBYLL_length_selection = []
    SIBYLL_width_selection = []
    final_SIBYLL_intens = []
    final_SIBYLL_length = []
    final_SIBYLL_width = []
    final_SIBYLL_pos = []

    for i in range(0, len(p_position)):
        QGS_intensity_selection_i = QGS_intensity.T[i] != 0
        QGS_length_selection_i = QGS_length.T[i] != 0
        QGS_width_selection_i = QGS_width.T[i] != 0
        SIBYLL_intensity_selection_i = SIBYLL_intensity.T[i] != 0
        SIBYLL_length_selection_i = SIBYLL_length.T[i] != 0
        SIBYLL_width_selection_i = SIBYLL_width.T[i] != 0
        if i == 0:
            QGS_intensity_selection = QGS_intensity_selection_i
            QGS_length_selection = QGS_length_selection_i
            QGS_width_selection = QGS_width_selection_i
            SIBYLL_intensity_selection = SIBYLL_intensity_selection_i
            SIBYLL_length_selection = SIBYLL_length_selection_i
            SIBYLL_width_selection = SIBYLL_width_selection_i
        else:
            QGS_intensity_selection = np.concatenate((QGS_intensity_selection,QGS_intensity_selection_i))
            QGS_length_selection = np.concatenate((QGS_length_selection, QGS_length_selection_i))
            QGS_width_selection = np.concatenate((QGS_width_selection, QGS_width_selection_i))
            SIBYLL_intensity_selection = np.concatenate((SIBYLL_intensity_selection,SIBYLL_intensity_selection_i))
            SIBYLL_length_selection = np.concatenate((SIBYLL_length_selection, SIBYLL_length_selection_i))
            SIBYLL_width_selection = np.concatenate((SIBYLL_width_selection, SIBYLL_width_selection_i))

        i_pos2 = np.repeat(p_position[i], QGS_intensity.T[i].size)
        final_QGS_pos = np.append(final_QGS_pos, i_pos2)
        i_pos3 = np.repeat(p_position[i], SIBYLL_intensity.T[i].size)
        final_SIBYLL_pos = np.append(final_SIBYLL_pos, i_pos3)

        final_QGS_intens = np.append(final_QGS_intens, QGS_intensity.T[i])
        final_QGS_length = np.append(final_QGS_length, QGS_length.T[i])
        final_QGS_width = np.append(final_QGS_width, QGS_width.T[i])
        final_SIBYLL_intens = np.append(final_SIBYLL_intens, SIBYLL_intensity.T[i])
        final_SIBYLL_length = np.append(final_SIBYLL_length, SIBYLL_length.T[i])
        final_SIBYLL_width = np.append(final_SIBYLL_width, SIBYLL_width.T[i])
else:
    pass


EPOS_exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], final_EPOS_pos[EPOS_length_selection], np.log10(final_EPOS_intens[EPOS_length_selection]))
EPOS_exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2], final_EPOS_pos[EPOS_length_selection], np.log10(final_EPOS_intens[EPOS_length_selection]))
EPOS_exp_width_interpol = find_exp_value(width_hist, width_w[1], width_w[2], final_EPOS_pos[EPOS_width_selection], np.log10(final_EPOS_intens[EPOS_width_selection]))
EPOS_exp_std_width_interpol = find_exp_value(stdWidthapprox, width_w[1], width_w[2], final_EPOS_pos[EPOS_width_selection], np.log10(final_EPOS_intens[EPOS_width_selection]))

EPOS_SCL = (final_EPOS_length[EPOS_length_selection] - EPOS_exp_length_interpol) / EPOS_exp_std_length_interpol
EPOS_SCW = (final_EPOS_width[EPOS_width_selection] - EPOS_exp_width_interpol) / EPOS_exp_std_width_interpol

# Calculates SC/W for QGS and SIBYLL
if compareProton == True:
    QGS_exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], final_QGS_pos[QGS_length_selection],np.log10(final_QGS_intens[QGS_length_selection]))
    QGS_exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2],final_QGS_pos[QGS_length_selection],np.log10(final_QGS_intens[QGS_length_selection]))
    QGS_exp_width_interpol = find_exp_value(width_hist, width_w[1], width_w[2], final_QGS_pos[QGS_width_selection],np.log10(final_QGS_intens[QGS_width_selection]))
    QGS_exp_std_width_interpol = find_exp_value(stdWidthapprox, width_w[1], width_w[2],final_QGS_pos[QGS_width_selection],np.log10(final_QGS_intens[QGS_width_selection]))

    SIBYLL_exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], final_SIBYLL_pos[SIBYLL_length_selection],np.log10(final_SIBYLL_intens[SIBYLL_length_selection]))
    SIBYLL_exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2],final_SIBYLL_pos[SIBYLL_length_selection],np.log10(final_SIBYLL_intens[SIBYLL_length_selection]))
    SIBYLL_exp_width_interpol = find_exp_value(width_hist, width_w[1], width_w[2], final_SIBYLL_pos[SIBYLL_width_selection],np.log10(final_SIBYLL_intens[SIBYLL_width_selection]))
    SIBYLL_exp_std_width_interpol = find_exp_value(stdWidthapprox, width_w[1], width_w[2],final_SIBYLL_pos[SIBYLL_width_selection],np.log10(final_SIBYLL_intens[SIBYLL_width_selection]))

    QGS_SCL = (final_QGS_length[QGS_length_selection] - QGS_exp_length_interpol) / QGS_exp_std_length_interpol
    QGS_SCW = (final_QGS_width[QGS_width_selection] - QGS_exp_width_interpol) / QGS_exp_std_width_interpol
    SIBYLL_SCL = (final_SIBYLL_length[SIBYLL_length_selection] - SIBYLL_exp_length_interpol) / SIBYLL_exp_std_length_interpol
    SIBYLL_SCW = (final_SIBYLL_width[SIBYLL_width_selection] - SIBYLL_exp_width_interpol) / SIBYLL_exp_std_width_interpol
else:
    pass

# Calculates SCL at individual Telescopes and compares the distributions' shift
IndividualTelescopes = False
if IndividualTelescopes == True:
    EPOS0_length_selection = EPOS_length.T[0] != 0
    EPOS0_pos = np.repeat(p_position[0], EPOS0_length_selection.size)
    EPOS0_exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], EPOS0_pos[EPOS0_length_selection], np.log10(EPOS_intensity.T[0][EPOS0_length_selection]))
    EPOS0_exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2], EPOS0_pos[EPOS0_length_selection], np.log10(EPOS_intensity.T[0][EPOS0_length_selection]))
    EPOS_SCL0 = (EPOS_length.T[0][EPOS0_length_selection] - EPOS0_exp_length_interpol) / EPOS0_exp_std_length_interpol

    EPOS120_length_selection = EPOS_length.T[3] != 0
    EPOS120_pos = np.repeat(p_position[3], EPOS120_length_selection.size)
    EPOS120_exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], EPOS120_pos[EPOS120_length_selection], np.log10(EPOS_intensity.T[3][EPOS120_length_selection]))
    EPOS120_exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2], EPOS120_pos[EPOS120_length_selection], np.log10(EPOS_intensity.T[3][EPOS120_length_selection]))
    EPOS_SCL120 = (EPOS_length.T[3][EPOS120_length_selection] - EPOS120_exp_length_interpol) / EPOS120_exp_std_length_interpol

    EPOS200_length_selection = EPOS_length.T[5] != 0
    EPOS200_pos = np.repeat(p_position[5], EPOS200_length_selection.size)
    EPOS200_exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], EPOS200_pos[EPOS200_length_selection], np.log10(EPOS_intensity.T[5][EPOS200_length_selection]))
    EPOS200_exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2], EPOS200_pos[EPOS200_length_selection], np.log10(EPOS_intensity.T[5][EPOS200_length_selection]))
    EPOS_SCL200 = (EPOS_length.T[5][EPOS200_length_selection] - EPOS200_exp_length_interpol) / EPOS200_exp_std_length_interpol

    EPOS320_length_selection = EPOS_length.T[8] != 0
    EPOS320_pos = np.repeat(p_position[8], EPOS320_length_selection.size)
    EPOS320_exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], EPOS320_pos[EPOS320_length_selection], np.log10(EPOS_intensity.T[8][EPOS320_length_selection]))
    EPOS320_exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2], EPOS320_pos[EPOS320_length_selection], np.log10(EPOS_intensity.T[8][EPOS320_length_selection]))
    EPOS_SCL320 = (EPOS_length.T[8][EPOS320_length_selection] - EPOS320_exp_length_interpol) / EPOS320_exp_std_length_interpol

    EPOS400_length_selection = EPOS_length.T[10] != 0
    EPOS400_pos = np.repeat(p_position[10], EPOS400_length_selection.size)
    EPOS400_exp_length_interpol = find_exp_value(length_hist, length_w[1], length_w[2], EPOS400_pos[EPOS400_length_selection], np.log10(EPOS_intensity.T[10][EPOS400_length_selection]))
    EPOS400_exp_std_length_interpol = find_exp_value(stdLengthapprox, length_w[1], length_w[2], EPOS400_pos[EPOS400_length_selection], np.log10(EPOS_intensity.T[10][EPOS400_length_selection]))
    EPOS_SCL400 = (EPOS_length.T[10][EPOS400_length_selection] - EPOS400_exp_length_interpol) / EPOS400_exp_std_length_interpol

    plt.clf()
    plt.hist(EPOS_SCL0, bins=100, range=(-3, 6), weights=np.zeros_like(EPOS_SCL0) + 1. / EPOS_SCL0.size, label='at 0m',
             histtype='step', alpha=1, color='red')
    plt.hist(EPOS_SCL120, bins=100, range=(-3, 6), weights=np.zeros_like(EPOS_SCL120) + 1. / EPOS_SCL120.size,
             label='at 120m', histtype='step', alpha=1, color='blue')
    plt.hist(EPOS_SCL200, bins=100, range=(-3, 6), weights=np.zeros_like(EPOS_SCL200) + 1. / EPOS_SCL200.size,
             label='at 200m', histtype='step', alpha=1, color='green')
    plt.hist(EPOS_SCL320, bins=100, range=(-3, 6), weights=np.zeros_like(EPOS_SCL320) + 1. / EPOS_SCL320.size,
             label='at 320m', histtype='step', alpha=1, color='yellow')
    plt.hist(EPOS_SCL400, bins=100, range=(-3, 6), weights=np.zeros_like(EPOS_SCL400) + 1. / EPOS_SCL400.size,
             label='at 400m', histtype='step', alpha=1, color='black')
    plt.xlabel('SCL')
    plt.ylabel('Event Fraction')
    plt.xticks(np.linspace(-3, 6, 10))
    plt.yticks(np.linspace(0, 0.07, 8))
    plt.grid(linestyle='--', linewidth=0.3)
    plt.legend()
    plt.title('EPOS SCL Distribution at different telescopes')
else:
    pass

# Binning by energies. mostly unneccessary now
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

# Do Proton Comparison Plots
if compareProton == True:
    avg_EPOS_l = calc_tel_avg(EPOS_length)
    avg_EPOS_w = calc_tel_avg(EPOS_width)
    avg_QGS_l = calc_tel_avg(QGS_length)
    avg_QGS_w = calc_tel_avg(QGS_width)
    avg_SIBYLL_l = calc_tel_avg(SIBYLL_length)
    avg_SIBYLL_w = calc_tel_avg(SIBYLL_width)

    HP_comp_plot(avg_EPOS_l,avg_QGS_l,avg_SIBYLL_l, 'Length')
    HP_comp_plot(avg_EPOS_w,avg_QGS_w,avg_SIBYLL_w, 'Width')

else:
    pass

SCL_cutoff_param = 1.0
SCW_cutoff_param = 0.7

EPOS_SCL_cutoff =  EPOS_SCL <= SCL_cutoff_param
EPOS_SCW_cutoff =  EPOS_SCW <= SCW_cutoff_param
EPOS_SCL_passing = EPOS_SCL[EPOS_SCL_cutoff].size/EPOS_length.size
EPOS_SCW_passing = EPOS_SCW[EPOS_SCW_cutoff].size/EPOS_width.size
if compareProton == True:
    QGS_SCL_cutoff = QGS_SCL <= SCL_cutoff_param
    QGS_SCW_cutoff = QGS_SCW <= SCW_cutoff_param
    QGS_SCL_passing = QGS_SCL[QGS_SCL_cutoff].size / QGS_length.size
    QGS_SCW_passing = QGS_SCW[QGS_SCW_cutoff].size / QGS_width.size

    SIBYLL_SCL_cutoff = SIBYLL_SCL <= SCL_cutoff_param
    SIBYLL_SCW_cutoff = SIBYLL_SCW <= SCW_cutoff_param
    SIBYLL_SCL_passing = SIBYLL_SCL[SIBYLL_SCL_cutoff].size / SIBYLL_length.size
    SIBYLL_SCW_passing = SIBYLL_SCW[SIBYLL_SCW_cutoff].size / SIBYLL_width.size

else:
    pass

# useful for seeing the distribution of the SCW to determine if gaussian
plt.clf()
plt.hist(EPOS_SCL, bins=100, range=(-3,6) ,weights=np.zeros_like(EPOS_SCL) + 1. / EPOS_SCL.size, label=str(filename_p1), histtype='step', alpha=1, color='red')
plt.hist(QGS_SCL, bins=100, range=(-3,6) ,weights=np.zeros_like(QGS_SCL) + 1. / QGS_SCL.size, label=str(filename_p2), histtype='step', alpha=1, color='green')
plt.hist(SIBYLL_SCL, bins=100, range=(-3,6) ,weights=np.zeros_like(SIBYLL_SCL) + 1. / SIBYLL_SCL.size, label=str(filename_p3), histtype='step', alpha=1, color='blue')
plt.hist(gamma_SCL, bins=100, range=(-3,6),weights=np.zeros_like(gamma_SCL) + 1. / gamma_SCL.size, label='Gamma', histtype='step', alpha=1, color='black')
plt.vlines(SCL_cutoff_param, 0, 0.07, linestyles='dashed', label='Cutoff')
plt.xlabel('SCL')
plt.ylabel('Event Fraction')
plt.xticks(np.linspace(-3,6,10))
plt.yticks(np.linspace(0,0.07,8))
plt.grid(linestyle = '--', linewidth = 0.3)
plt.legend()
plt.title('SCL Distribution')

endtime = time.time()
print('running took', endtime-starttime, 'seconds')

#print('Now writing:' + "Schmutz_test2.txt")
#textfile = open(save_dir + "Schmutz_test2.txt", "w+")
#
#textfile.close()
