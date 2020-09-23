import uproot
import corsika_toy_iact.iact_array as iact
import numpy as np
import pkg_resources
import matplotlib.pyplot as plt

positions = np.array([[0, 0], [0, 20], [0, 40], [0, 60]])
iact_array = iact.IACTArray(positions)
data_dir = pkg_resources.resource_filename('corsika_toy_iact', 'data/')

image = iact_array.process_file_list([data_dir+"test_data.root"])

image_pe = iact_array.scale_to_photoelectrons(psf_width=0.05, mirror_reflectivity=0.8, pedestal_width=0.8)

for image in image_pe:
    plt.imshow(image[0])
    plt.colorbar()
    plt.show()