import uproot
import corsika_toy_iact.iact_array as iact
import numpy as np
import pkg_resources
import matplotlib.pyplot as plt
from ctapipe.image import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError

x = np.linspace(-180, 180, 4)
xx, yy = np.array(np.meshgrid(x, x))
positions = np.vstack((xx.ravel(), yy.ravel())).T

iact_array = iact.IACTArray(positions, radius=6)
data_dir = pkg_resources.resource_filename('corsika_toy_iact', 'data/')

#image = iact_array.process_file_list([data_dir+"test_data.root"])
header, image = iact_array.process_file_list(["corsika_toy_iact/data/CER000001.root"])

image_pe = iact_array.scale_to_photoelectrons(psf_width=0.04, mirror_reflectivity=0.8,
                                              quantum_efficiency=0.2, pedestal_width=0.8,
                                              single_pe_width=0.5, photon_cut=10)

geometry = iact_array.get_camera_geometry()
i, j = 0, 0
width = np.zeros((image_pe.shape[0], image_pe.shape[1]))

for event in image_pe:
    for image in event:
        mask = tailcuts_clean(geometry, image.ravel(),
                              picture_thresh=8, boundary_thresh=4).reshape(40, 40)
        image[np.invert(mask)] = 0
        try:
            hill = hillas_parameters(geometry, image.ravel())
            width[i][j] = hill.intensity
            j += 1
        except HillasParameterizationError:
            j += 1

    i += 1
    j = 0

print(np.sum(header["energy"] > 10000), np.sum(width[header["energy"] > 10000].ravel() > 0))
plt.hist(width[header["energy"] > 10000].ravel(), bins=50, log=True)

plt.show()
