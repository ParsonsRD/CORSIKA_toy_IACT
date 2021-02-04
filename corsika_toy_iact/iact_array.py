import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from numpy.random import normal, poisson
import uproot4 as uproot
from ctapipe.coordinates import AltAz, NominalFrame
import astropy.units as units

"""
This class is created to process to output from CORSIKA Cherenkov output files (after conversion to ROOT TTrees)
This output can then be read and processed to simply simulate the behaviour of IACTs. This is done by simply binning 
the directions of the photons landing on a region of the ground in a histogram. Then the photon positions can be 
smoothed to represent the effect of the optical PSF, then efficiency factors and noise added. This effectively converts
this number of photons to a number of photoelectrons as typically measured in a Cherenkov telescope.
"""


class IACTArray:

    def __init__(self, telescope_positions, radius=5, camera_size=8, bins=40, multiple_cores=False):
        """
        :param telescope_positions: ndarray
            X, Y positions of telescopes of ground (N, 2)
        :param radius: float
            Radius of telescope mirrors (m)
        :param camera_size: float
            Size of camera (deg)
        :param bins: int
            Number of pixels on each camera axis
        """

        # Just copy everything into this class that we need
        telescope_positions = np.array(telescope_positions)
        self.telescope_x_positions = telescope_positions.T[0]
        self.telescope_y_positions = telescope_positions.T[1]

        self.telescope_radius = radius
        self.camera_radius = camera_size/2.
        self.camera_axis_bins = bins

        self.num_telescope = telescope_positions.shape[0]
        self.images = None
        self.multiple_cores = multiple_cores

    @staticmethod
    def _get_event_numbers(photon_list):
        """
        Gets a list of unique event numbers in the file read it

        :param photon_list: dict
            Dictionary of the Cherenkov ROOT file contents
        :return: ndarray
            Array of unique event numbers
        """
        events = photon_list["event"]
        return np.unique(events)

    def process_images(self, header, photon_list):
        """
        Read in the list of photons and create a binned camera image for all defined telescopes
        and all events

        :param photon_list: dict
            Dictionary of the Cherenkov ROOT file contents
        :return: ndarray
            4D array of camera images for all events and telescopes
        """

        # First get our list of unique event numbers
        event_nums = header["event"]#self._get_event_numbers(header)
        print(event_nums)
        image_list = []
        event_count = 0
        array_pointing = AltAz(alt=90*units.deg, az=0*units.deg)

        print('Processing images')
        previous_event = -1
        # Then loop over the events
        for event in tqdm(event_nums):

            if self.multiple_cores:
                event_base = int(np.floor(event / 100.))
            else:
                event_base = event

            # Select the photons from the list belonging to this event
            selection = photon_list["event"] == event_base

            if event_base != previous_event:
                # Photons positions in m
                x = photon_list["x"][selection]/100
                y = photon_list["y"][selection]/100

                # Photons directions in deg
                u = photon_list["u"][selection] * (180/np.pi)
                v = photon_list["v"][selection] * (180/np.pi)
                #print(u)
                # Weight of each photon
                weights = photon_list["s"][selection]
                previous_event = event_base

                if self.multiple_cores:
                    core_x = header["core_x"][event_count]
                    core_y = header["core_y"][event_count]
                else:
                    core_x = 0
                    core_y = 0

            # Calculate which telescope each photon belongs to
            r = np.sqrt(np.power(x[:, np.newaxis] - (self.telescope_x_positions + core_x), 2) +
                        np.power(y[:, np.newaxis] - (self.telescope_y_positions + core_y), 2))

            telescope_selection = np.array(r < self.telescope_radius, np.int)

            tel_list = np.arange(1, self.num_telescope+1)
            telescope_selection *= tel_list[np.newaxis, :]

            telescope_seen = np.sum(telescope_selection, axis=1)

            # Then bin our photons
            content, edges = np.histogramdd((telescope_seen, u, v),
                                            bins=(self.num_telescope, self.camera_axis_bins, self.camera_axis_bins),
                                            range=((0.5, self.num_telescope+0.5),
                                                   (-self.camera_radius, self.camera_radius),
                                                   (-self.camera_radius, self.camera_radius)),
                                            weights=weights)
            # Add this event onto our list of events
            image_list.append(content)
            event_count = event_count + 1

        # And add ont our array for all files
        image_list = np.array(image_list)
        print("s", np.sum(image_list[0][0]))
        if self.images is None:
            self.images = image_list
        else:
            self.images = np.append(self.images, image_list, 0)

        return image_list

    def process_file_list(self, file_list, max_events=1e10):
        """
        Read in and bin photons from a list of CORSIKA output ROOT files

        :param file_list: list
            List of files to read in
        """
        self.reset()
        header_array = None
        for file in file_list:
            print('Reading', file)

            events = uproot.open(file)

            branches = events["photons"].arrays(library="np")
            header = events["header"].arrays(library="np", entry_stop=max_events)

            if header_array is None:
                header_array = header
            else:
                header_array = np.concatenate(header_array, header, axis=0)
            self.process_images(header, branches)

        return header_array, self.images.astype(np.float32)

    def _apply_optical_psf(self, images, psf_width, **kwargs):
        """
        Smooth the images with the a gaussian kernel to represent the PSF of the telescope
        :param images: ndarray
            Array of camera images
        :param psf_width: float
            Width of gaussian PSF
        :return: ndarray
            Smoothed camera images
        """
        psf_width = psf_width * (self.camera_axis_bins / (self.camera_radius * 2))
        return gaussian_filter(images, sigma=(0, 0, psf_width, psf_width))

    @staticmethod
    def _apply_efficiency(images, mirror_reflectivity=0.8, quantum_efficiency=0.2,
                          single_pe_width=0.5, pedestal_width=1, **kwargs):
        """
        Apply scaling factor to simulate the efficiency of the mirrors and photodetectors. Add random gaussian
        pedestal values

        :param images: ndarray
            Array of camera images
        :param mirror_reflectivity: float
            Efficiency of mirrors
        :param quantum_efficiency: float
            Efficiency of photodetectors
        :param pedestal_width: float
            Width of gaussian pedestal to add
        :return:
        """
        scaled_images = images * mirror_reflectivity * quantum_efficiency
        pedestal = normal(0, pedestal_width, images.shape)

        # Should we round here for Poisson?
        scaled_images = normal(scaled_images, np.sqrt(scaled_images) * single_pe_width)
        return scaled_images + pedestal

    @staticmethod
    def _apply_photon_cut(images, photon_cut=10, **kwargs):
        """

        :param images:
        :param photon_cut:
        :param kwargs:
        :return:
        """
        flatten_image = images.reshape(images.shape[0], images.shape[1], images.shape[2] * images.shape[3])
        image_sum = np.sum(flatten_image, axis=-1)

        image_mask = image_sum > photon_cut
        return image_mask[:, :, np.newaxis, np.newaxis]

    def scale_to_photoelectrons(self, **kwargs):
        """
        Apply both efficiency smoothing in one step
        :return: ndarray
            Images now converted to photoelectrons
        """

        smoothed_images = self._apply_optical_psf(self.images, **kwargs)
        return self._apply_efficiency(smoothed_images, **kwargs) * self._apply_photon_cut(self.images, **kwargs)

    def get_camera_geometry(self):
        """
        Get ctapipe style camera object for our simulated camera type.
        Object returns results in m, but can ignore this.
        :return: CameraGeometry
            Camera geometry object
        """
        from ctapipe.instrument import CameraGeometry
        half_pixel_size = self.camera_radius/float(self.camera_axis_bins)
        radius = self.camera_radius-half_pixel_size
        return CameraGeometry.make_rectangular(self.camera_axis_bins, self.camera_axis_bins,
                                               range_x=(-radius, radius),
                                               range_y=(-radius,  radius))

    def reset(self):
        self.images = None
