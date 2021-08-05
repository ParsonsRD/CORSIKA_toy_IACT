import numpy as np
import math
import scipy.constants as const
import matplotlib.pyplot as plt
import pkg_resources

def readfile(datafile):
    '''Input: .txt datafile you want to read in
        Output: array where 0th component is avg_intensity, 1st ist avg_length, 2nd is avg_width, 3rd is tenth_percentile_intensity and so on'''
    bigarray = np.genfromtxt(datafile, comments=None,dtype=str)
    ind = np.arange(0, bigarray.size, 1)
    res = {}
    marker = ind[np.char.startswith(bigarray, '#')]
    for i in range(len(marker)):

        if i == len(marker) - 1:
            res[bigarray[marker[i]]] = np.asarray(bigarray[marker[i] + 1:], dtype=float)
            break
        res[bigarray[marker[i]]] = np.asarray(bigarray[marker[i] + 1: marker[i + 1]], dtype=float)

    avg_intensity = res['#average_intensity']
    avg_length = res['#average_length']
    avg_width = res['#average_width']
    std_intensity = res['#std_intensity']
    std_length = res['#std_length']
    std_width = res['#std_width']
    intensity = res['#intensity']
    length = res['#length']
    width = res['#width']
    energy = res['#energy']
    impact = res['#impact_dist']
    intensity_hist = res['#intensity_hist']
    length_hist = res['#length_hist']
    width_hist = res['#width_hist']

    #tenth_percentile_intensity = res['#tenth_percentile_intensity']
    #tenth_percentile_length = res['#tenth_percentile_length']
    #tenth_percentile_width = res['#tenth_percentile_width']

    return res #avg_intensity, avg_length, avg_width, std_intensity, std_length, std_width, intensity, length, width

def naming(fname, data_array): #unfinished
    '''
    Input:
        fname: the prefix you want to give the variables, e.g. EPOS_10TeV_proton
        data_array = array where 0th component is avg_intensity, 1st ist avg_length, 2nd is avg_width, 3rd is tenth_percentile_intensity and so on
    Output:
        renamed variables that map to the according array, e.g. EPOS_10TeV_proton_average_intensity = avg_intensity_array
        '''
    gname = str(fname)
    return


def make_comparison_plots(xlist,ylist, label, title): #for like proton data
    '''
    :param xlist: array, e.g. telescope_position
    :param ylist: list of arrays, e.g. [avgerage_intensity_EPOS, average_intensity_QGSII]
    :param label: list of strings, e.g. ['EPOS data, QGSII data']
    :param title: string, e.g. 'Plot of average intensities'
    :return: Plot
    '''
    for i in range(0,len(ylist)):
        plt.plot(xlist,ylist[i], marker='x', ls='--', label=label[i])
    title=str(title)
    plt.title(title)
    plt.xlabel('Telescope position in m')
    if 'Intensities' in title or 'intensities' in title or 'Intensity' in title or 'intensity' in title:
        #plt.yscale('log')
        plt.ylabel('Intensity in photoelectrons')
    if 'Length' in title or 'length' in title:
        plt.ylabel('Length in degrees')
    if 'Width' in title or 'width' in title:
        plt.ylabel('Width in degrees')
    plt.legend()
    plt.grid()
    #plt.savefig(plot_save_dir+title+'.pdf')
    plt.show()
    return

#Import Data Sets
telescope_pos_20 = np.linspace(0, 400, 21)
save_dir = pkg_resources.resource_filename('corsika_toy_iact', 'save_dir/txt_and_jpg/')
plot_save_dir = pkg_resources.resource_filename('corsika_toy_iact', 'save_dir/comparison_plots/')

EPOS_100GeV_proton_data   = save_dir+"./EPOS_100GeV_proton.root.txt"
EPOS_1TeV_proton_data     = save_dir+"./EPOS_1000GeV_proton.root.txt"
EPOS_10TeV_proton_data    = save_dir+"./EPOS_10000GeV_proton.root.txt"
QGSII_100GeV_proton_data  = save_dir+"./QGSII_100GeV_proton.root.txt"
QGSII_1TeV_proton_data    = save_dir+"./QGSII_1000GeV_proton.root.txt"
QGSII_10TeV_proton_data   = save_dir+"./QGSII_10000GeV_proton.root.txt"
SIBYLL_100GeV_proton_data = save_dir+"./SIBYLL_100GeV_proton.root.txt"
SIBYLL_1TeV_proton_data   = save_dir+"./SIBYLL_1000GeV_proton.root.txt"
SIBYLL_10TeV_proton_data  = save_dir+"./SIBYLL_10000GeV_proton.root.txt"
gamma_example_data  = save_dir+"./gamma_example.root.txt"
gamma001 = save_dir+"./CER000101.root.txt"


# 100 GeV

percentile_intensity_100GeV = [readfile(EPOS_100GeV_proton_data)[4],readfile(QGSII_100GeV_proton_data)[4], readfile(SIBYLL_100GeV_proton_data)[4]]
intensity_over_EPOS_100GeV = [[a/b for a,b in zip(readfile(EPOS_100GeV_proton_data)[0],readfile(EPOS_100GeV_proton_data)[0])],
                              [a/b for a,b in zip(readfile(QGSII_100GeV_proton_data)[0],readfile(EPOS_100GeV_proton_data)[0])],
                              [a/b for a,b in zip(readfile(SIBYLL_100GeV_proton_data)[0],readfile(EPOS_100GeV_proton_data)[0])]]
length_over_EPOS_100GeV    = [[a/b for a,b in zip(readfile(EPOS_100GeV_proton_data)[1],readfile(EPOS_100GeV_proton_data)[1])],
                              [a/b for a,b in zip(readfile(QGSII_100GeV_proton_data)[1],readfile(EPOS_100GeV_proton_data)[1])],
                              [a/b for a,b in zip(readfile(SIBYLL_100GeV_proton_data)[1],readfile(EPOS_100GeV_proton_data)[1])]]
width_over_EPOS_100GeV     = [[a/b for a,b in zip(readfile(EPOS_100GeV_proton_data)[2],readfile(EPOS_100GeV_proton_data)[2])],
                              [a/b for a,b in zip(readfile(QGSII_100GeV_proton_data)[2],readfile(EPOS_100GeV_proton_data)[2])],
                              [a/b for a,b in zip(readfile(SIBYLL_100GeV_proton_data)[2],readfile(EPOS_100GeV_proton_data)[2])]]

percentile_intensity_over_EPOS_100GeV = [[a/b for a,b in zip(readfile(EPOS_100GeV_proton_data)[3],readfile(EPOS_100GeV_proton_data)[3])],
                                        [a/b for a,b in zip(readfile(QGSII_100GeV_proton_data)[3],readfile(EPOS_100GeV_proton_data)[3])],
                                        [a/b for a,b in zip(readfile(SIBYLL_100GeV_proton_data)[3],readfile(EPOS_100GeV_proton_data)[3])]]
percentile_length_over_EPOS_100GeV =    [[a/b for a,b in zip(readfile(EPOS_100GeV_proton_data)[4],readfile(EPOS_100GeV_proton_data)[4])],
                                        [a/b for a,b in zip(readfile(QGSII_100GeV_proton_data)[4],readfile(EPOS_100GeV_proton_data)[4])],
                                        [a/b for a,b in zip(readfile(SIBYLL_100GeV_proton_data)[4],readfile(EPOS_100GeV_proton_data)[4])]]
percentile_width_over_EPOS_100GeV =      [[a/b for a,b in zip(readfile(EPOS_100GeV_proton_data)[5],readfile(EPOS_100GeV_proton_data)[5])],
                                        [a/b for a,b in zip(readfile(QGSII_100GeV_proton_data)[5],readfile(EPOS_100GeV_proton_data)[5])],
                                        [a/b for a,b in zip(readfile(SIBYLL_100GeV_proton_data)[5],readfile(EPOS_100GeV_proton_data)[5])]]

# 1 TeV

percentile_intensity_1TeV = [readfile(EPOS_1TeV_proton_data)[4],readfile(QGSII_1TeV_proton_data)[4], readfile(SIBYLL_1TeV_proton_data)[4]]
intensity_over_EPOS_1TeV = [[a/b for a,b in zip(readfile(EPOS_1TeV_proton_data)[0],readfile(EPOS_1TeV_proton_data)[0])],
                             [a/b for a,b in zip(readfile(QGSII_1TeV_proton_data)[0],readfile(EPOS_1TeV_proton_data)[0])],
                             [a/b for a,b in zip(readfile(SIBYLL_1TeV_proton_data)[0],readfile(EPOS_1TeV_proton_data)[0])]]
length_over_EPOS_1TeV    = [[a/b for a,b in zip(readfile(EPOS_1TeV_proton_data)[1],readfile(EPOS_1TeV_proton_data)[1])],
                             [a/b for a,b in zip(readfile(QGSII_1TeV_proton_data)[1],readfile(EPOS_1TeV_proton_data)[1])],
                             [a/b for a,b in zip(readfile(SIBYLL_1TeV_proton_data)[1],readfile(EPOS_1TeV_proton_data)[1])]]
width_over_EPOS_1TeV = [[a/b for a,b in zip(readfile(EPOS_1TeV_proton_data)[2],readfile(EPOS_1TeV_proton_data)[2])],
                             [a/b for a,b in zip(readfile(QGSII_1TeV_proton_data)[2],readfile(EPOS_1TeV_proton_data)[2])],
                             [a/b for a,b in zip(readfile(SIBYLL_1TeV_proton_data)[2],readfile(EPOS_1TeV_proton_data)[2])]]

percentile_intensity_over_EPOS_1TeV = [[a/b for a,b in zip(readfile(EPOS_1TeV_proton_data)[3],readfile(EPOS_1TeV_proton_data)[3])],
                                        [a/b for a,b in zip(readfile(QGSII_1TeV_proton_data)[3],readfile(EPOS_1TeV_proton_data)[3])],
                                        [a/b for a,b in zip(readfile(SIBYLL_1TeV_proton_data)[3],readfile(EPOS_1TeV_proton_data)[3])]]
percentile_length_over_EPOS_1TeV =    [[a/b for a,b in zip(readfile(EPOS_1TeV_proton_data)[4],readfile(EPOS_1TeV_proton_data)[4])],
                                        [a/b for a,b in zip(readfile(QGSII_1TeV_proton_data)[4],readfile(EPOS_1TeV_proton_data)[4])],
                                        [a/b for a,b in zip(readfile(SIBYLL_1TeV_proton_data)[4],readfile(EPOS_1TeV_proton_data)[4])]]
percentile_width_over_EPOS_1TeV =      [[a/b for a,b in zip(readfile(EPOS_1TeV_proton_data)[5],readfile(EPOS_1TeV_proton_data)[5])],
                                        [a/b for a,b in zip(readfile(QGSII_1TeV_proton_data)[5],readfile(EPOS_1TeV_proton_data)[5])],
                                        [a/b for a,b in zip(readfile(SIBYLL_1TeV_proton_data)[5],readfile(EPOS_1TeV_proton_data)[5])]]

#10 TeV

percentile_intensity_10TeV = [readfile(EPOS_10TeV_proton_data)[4],readfile(QGSII_10TeV_proton_data)[4], readfile(SIBYLL_10TeV_proton_data)[4]]
intensity_over_EPOS_10TeV = [[a/b for a,b in zip(readfile(EPOS_10TeV_proton_data)[0],readfile(EPOS_10TeV_proton_data)[0])],
                             [a/b for a,b in zip(readfile(QGSII_10TeV_proton_data)[0],readfile(EPOS_10TeV_proton_data)[0])],
                             [a/b for a,b in zip(readfile(SIBYLL_10TeV_proton_data)[0],readfile(EPOS_10TeV_proton_data)[0])]]
length_over_EPOS_10TeV    = [[a/b for a,b in zip(readfile(EPOS_10TeV_proton_data)[1],readfile(EPOS_10TeV_proton_data)[1])],
                             [a/b for a,b in zip(readfile(QGSII_10TeV_proton_data)[1],readfile(EPOS_10TeV_proton_data)[1])],
                             [a/b for a,b in zip(readfile(SIBYLL_10TeV_proton_data)[1],readfile(EPOS_10TeV_proton_data)[1])]]
width_over_EPOS_10TeV = [[a/b for a,b in zip(readfile(EPOS_10TeV_proton_data)[2],readfile(EPOS_10TeV_proton_data)[2])],
                             [a/b for a,b in zip(readfile(QGSII_10TeV_proton_data)[2],readfile(EPOS_10TeV_proton_data)[2])],
                             [a/b for a,b in zip(readfile(SIBYLL_10TeV_proton_data)[2],readfile(EPOS_10TeV_proton_data)[2])]]

percentile_intensity_over_EPOS_10TeV = [[a/b for a,b in zip(readfile(EPOS_10TeV_proton_data)[3],readfile(EPOS_10TeV_proton_data)[3])],
                                        [a/b for a,b in zip(readfile(QGSII_10TeV_proton_data)[3],readfile(EPOS_10TeV_proton_data)[3])],
                                        [a/b for a,b in zip(readfile(SIBYLL_10TeV_proton_data)[3],readfile(EPOS_10TeV_proton_data)[3])]]
percentile_length_over_EPOS_10TeV =    [[a/b for a,b in zip(readfile(EPOS_10TeV_proton_data)[4],readfile(EPOS_10TeV_proton_data)[4])],
                                        [a/b for a,b in zip(readfile(QGSII_10TeV_proton_data)[4],readfile(EPOS_10TeV_proton_data)[4])],
                                        [a/b for a,b in zip(readfile(SIBYLL_10TeV_proton_data)[4],readfile(EPOS_10TeV_proton_data)[4])]]
percentile_width_over_EPOS_10TeV =      [[a/b for a,b in zip(readfile(EPOS_10TeV_proton_data)[5],readfile(EPOS_10TeV_proton_data)[5])],
                                        [a/b for a,b in zip(readfile(QGSII_10TeV_proton_data)[5],readfile(EPOS_10TeV_proton_data)[5])],
                                        [a/b for a,b in zip(readfile(SIBYLL_10TeV_proton_data)[5],readfile(EPOS_10TeV_proton_data)[5])]]


#make_comparison_plots(telescope_pos_20, intensity_over_EPOS_10TeV, ['EPOS', 'QGSII', 'SIBYLL'], '10TeV 10th Percentile Width compared to EPOS')