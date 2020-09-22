import uproot
import corsika_toy_iact.iact_array as iact
import numpy as np

positions = np.array([[0, 0], [0, 20], [0, 40], [0, 60]])
iact_array = iact.IACTArray(positions)
image = iact_array.process_file_list(["data/epos_10TeV_cher.root"])

image_pe = iact_array.scale_to_photoelectrons(psf_width=0.05, mirror_reflectivity=0.8, pedestal_width=0.8)

import matplotlib.pyplot as plt
for image in image_pe:
    plt.imshow(image[0])
    plt.colorbar()
    plt.show()