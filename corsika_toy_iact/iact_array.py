import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from numpy.random import normal, poisson
import uproot
"""
This class is created to process to output from CORSIKA Cherenkov output files (after conversion to ROOT TTrees)
This output can then be read and processed to simply simulate the behaviour of IACTs. This is done by simply binning 
the directions of the photons landing on a region of the ground in a histogram. Then the photon positions can be 
smoothed to represent the effect of the optical PSF, then efficiency factors and noise added. This effectively converts
this number of photons to a number of photoelectrons as typically measured in a Cherenkov telescope.
"""


class IACTArray:

    def __init__(self, telescope_positions, radius=5, camera_size=4, bins=80):
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

    def process_images(self, photon_list):
        """
        Read in the list of photons and create a binned camera image for all defined telescopes
        and all events

        :param photon_list: dict
            Dictionary of the Cherenkov ROOT file contents
        :return: ndarray
            4D array of camera images for all events and telescopes
        """

        # First get our list of unique event numbers
        event_nums = self._get_event_numbers(photon_list)
        image_list = []

        print('Processing images')
        # Then loop over the events
        for event in tqdm(event_nums):

            # Select the photons from the list belonging to this event
            selection = photon_list["event"] == event
            # Photons positions in m
            x = photon_list["x"][selection]/100
            y = photon_list["y"][selection]/100

            # Photons directions in deg
            u = np.rad2deg(photon_list["u"][selection])
            v = np.rad2deg(photon_list["v"][selection])

            # Weight of each photon
            weights = photon_list["s"][selection]

            # Calculate which telescope each photon belongs to
            r = np.sqrt(np.power(x[:, np.newaxis] - self.telescope_x_positions, 2) +
                        np.power(y[:, np.newaxis] - self.telescope_y_positions, 2))
            telescope_selection = np.array(r < self.telescope_radius, np.int)
            tel_list = np.arange(1, self.num_telescope+1)
            telescope_selection *= tel_list[np.newaxis, :]

            telescope_seen = np.sum(telescope_selection, axis=1)

            # Then bin our photons
            content, edges = np.histogramdd((telescope_seen, u, v),
                                            bins=(self.num_telescope, self.camera_axis_bins, self.camera_axis_bins),
                                            range=((0.5, self.num_telescope+1.5),
                                                   (-self.camera_radius, self.camera_radius),
                                                   (-self.camera_radius, self.camera_radius)),
                                            weights=weights)
            # Add this event onto our list of events
            image_list.append(content)

        # And add ont our array for all files
        image_list = np.array(image_list)
        if self.images is None:
            self.images = image_list
        else:
            self.images = np.append(self.images, image_list, 0)

        return image_list

    def process_file_list(self, file_list):
        """
        Read in and bin photons from a list of CORSIKA output ROOT files

        :param file_list: list
            List of files to read in
        """

        for file in file_list:
            print('Reading', file)

            events = uproot.open(file)["photons"]
            branches = events.arrays(["event", "x", "y", "u", "v", "s"], namedecode="utf-8")
            self.process_images(branches)

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

        scaled_images = poisson(scaled_images)
        scaled_images = normal(scaled_images, single_pe_width * scaled_images)
        return scaled_images + pedestal

    def scale_to_photoelectrons(self, **kwargs):
        """
        Apply both efficiency smoothing in one step
        :return: ndarray
            Images now converted to photoelectrons
        """
        smoothed_images = self._apply_optical_psf(self.images, **kwargs)
        return self._apply_efficiency(smoothed_images, **kwargs)
