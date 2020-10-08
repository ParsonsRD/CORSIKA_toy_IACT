import uproot
import corsika_toy_iact.iact_array as iact
import numpy as np
import pkg_resources
import matplotlib.pyplot as plt
from ctapipe.image import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError

positions = np.array([[0, 0], [20, 0], [40, 0], [200, 0]])
iact_array = iact.IACTArray(positions)
data_dir = pkg_resources.resource_filename('corsika_toy_iact', 'data/')

#image = iact_array.process_file_list([data_dir+"test_data.root"])
image = iact_array.process_file_list(["corsika_toy_iact/data/epos_10TeV_cher.root"])

image_pe = iact_array.scale_to_photoelectrons(psf_width=0.04, mirror_reflectivity=0.8,
                                              quantum_efficiency=0.2, pedestal_width=0.8,
                                              single_pe_width=0.5)

geometry = iact_array.get_camera_geometry()
i, j = 0, 0
width = np.zeros((image_pe.shape[0], image_pe.shape[1]))

for event in image_pe:
    for image in event:
        mask = tailcuts_clean(geometry, image.ravel(),
                              picture_thresh=8, boundary_thresh=4).reshape(80, 80)
        image[np.invert(mask)] = 0
        try:
            hill = hillas_parameters(geometry, image.ravel())
            width[i][j] = hill.intensity
            j += 1
        except HillasParameterizationError:
            j += 1

    i += 1
    j = 0

plt.hist(width.T[3], bins=50)
plt.show()