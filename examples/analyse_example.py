import uproot
import corsika_toy_iact.iact_array as iact

events = uproot.open("data/epos_1TeV_cher.root")["photons"]
branches = events.arrays(["event","x","y","u","v","s"], namedecode="utf-8")
print(list(branches.keys())[0])

iact_array = iact.IACTArray([[0.,0.], [20.,0.]])
iact_array.process_images(branches)
