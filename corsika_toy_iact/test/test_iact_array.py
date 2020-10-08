import corsika_toy_iact.iact_array as iact
import pkg_resources
import uproot as uproot
import numpy as np

iact_array = iact.IACTArray([[0., 0.], [0., 20.]], radius=5, camera_size=1)
data_dir = pkg_resources.resource_filename('corsika_toy_iact', 'data/')
file = data_dir + "test_data.root"

events = uproot.open(file)["photons"]
branches = events.arrays(["event", "x", "y", "u", "v", "s"], namedecode="utf-8")


def test_event_numbers():
    iact_array.reset()
    event_list = iact_array._get_event_numbers(branches)
    assert np.allclose(event_list, np.arange(10)+1)


def test_reading():
    iact_array.reset()
    images = iact_array.process_images(branches)
    assert images.shape == (10, 2, 80, 80)


def test_sums():
    iact_array.reset()
    images = iact_array.process_images(branches)
    event_sel = branches["event"] == 1
    r = np.sqrt(branches["x"] * branches["x"] + branches["y"] * branches["y"])

    radius_sel = np.logical_and(r < 5 * 100, event_sel)
    fov_sel = np.logical_and(np.abs(np.rad2deg(branches["u"])) < 0.5, np.abs(np.rad2deg(branches["v"])) < 0.5)

    photon_sel = np.logical_and(radius_sel, fov_sel)
    assert np.sum(branches["s"][photon_sel]) == np.sum(images[0][0])


def test_read_list():
    iact_array.reset()
    images = iact_array.process_images(branches)

    iact_array_list = iact.IACTArray([[0., 0.], [0., 20.]], radius=5, camera_size=1)
    images_list = iact_array_list.process_file_list([file])
    assert np.allclose(images, images_list)

    images_list_2 = iact_array_list.process_file_list([file, file])

    assert images_list_2.shape[0] == 2 * images_list.shape[0]


def test_psf():
    iact_array.reset()
    images = iact_array.process_images(branches)
    smoothed_images = iact_array._apply_optical_psf(images, psf_width=0.05)
    assert np.allclose(np.sum(images), np.sum(smoothed_images), rtol=0.01)


def test_efficiency():
    iact_array.reset()
    images = iact_array.process_images(branches)
    scaled_images = iact_array._apply_efficiency(images, mirror_reflectivity=0.1, quantum_efficiency=0.1,
                                                 pedestal_width=0., single_pe_width=0.)
    print(np.sum(scaled_images), np.sum(images*0.01))

    # This is a bit iffy as we have random numbers, but given the p.e. and tolerance should return True
    # from this data if all is well
    assert np.allclose(np.sum(images * 0.01), np.sum(scaled_images), rtol=0.05)


def test_combined():
    iact_array.reset()
    images = iact_array.process_images(branches)
    scaled_images = iact_array.scale_to_photoelectrons(mirror_reflectivity=0.1, quantum_efficiency=0.1,
                                                       pedestal_width=0., single_pe_width=0., psf_width=0.05)

    assert np.allclose(np.sum(images * 0.01), np.sum(scaled_images), rtol=0.05)


def test_geometry():
    iact_array.reset()
    geom = iact_array.get_camera_geometry()
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 80), np.linspace(-0.5, 0.5, 80))

    assert np.allclose(geom.pix_x.value, x.ravel())
    assert np.allclose(geom.pix_y.value, y.ravel())
