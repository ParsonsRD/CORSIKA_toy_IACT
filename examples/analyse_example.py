import uproot
import corsika_toy_iact.iact_array as iact

iact_array = iact.IACTArray([[0.,0.], [20.,0.]])
image = iact_array.process_file_list(["data/epos_1TeV_cher.root"])

image_pe = iact_array.scale_to_photoelectrons(psf_width=0.05, mirror_reflectivity=0.8, pedestal_width=0.8)


