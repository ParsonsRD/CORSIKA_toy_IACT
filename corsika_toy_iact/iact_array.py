import numpy as np
from tqdm import tqdm
"""
"""


class IACTArray:

    def __init__(self, telescope_positions, radius=5, camera_size=4, bins=80):
        """

        :param telescope_positions:
        :param radius:
        :param camera_size:
        :param bins:
        """

        telescope_positions = np.array(telescope_positions)
        self.telescope_x_positions = telescope_positions.T[0]
        self.telescope_y_positions = telescope_positions.T[1]

        self.telescope_radius = radius
        self.camera_radius = camera_size/2.
        self.camera_axis_bins = bins

        self.num_telescope = telescope_positions.shape[0]
        self.images = None

    @staticmethod
    def get_event_numbers(photon_list):
        events = photon_list["event"]
        return np.unique(events)

    def process_images(self, photon_list):

        event_nums = self.get_event_numbers(photon_list)
        num_events = event_nums.shape[0]
        new_images = np.zeros((num_events, self.num_telescope, self.camera_axis_bins, self.camera_axis_bins))

        image_list = []

        print('Processing images')
        for event in tqdm(event_nums):

            selection = photon_list["event"] == event
            x = photon_list["x"][selection]/100
            y = photon_list["y"][selection]/100

            u = np.rad2deg(photon_list["u"][selection])
            v = np.rad2deg(photon_list["v"][selection])

            weights = photon_list["s"][selection]

            r = np.sqrt(np.power(x[:, np.newaxis] - self.telescope_x_positions, 2) +
                        np.power(y[:, np.newaxis] - self.telescope_y_positions, 2))
            telescope_selection = np.array(r < self.telescope_radius, np.int)
            tel_list = np.arange(1, self.num_telescope+1)
            telescope_selection *= tel_list[np.newaxis, :]

            telescope_seen = np.sum(telescope_selection, axis=1)

            content, edges = np.histogramdd((telescope_seen, u, v),
                                            bins=(self.num_telescope, self.camera_axis_bins, self.camera_axis_bins),
                                            range=((0.5, self.num_telescope+1.5),
                                                   (-self.camera_radius, self.camera_radius),
                                                   (-self.camera_radius, self.camera_radius)),
                                            weights=weights)
            image_list.append(content)

        image_list = np.array(image_list)
        if self.images is None:
            self.images = image_list
        else:
            self.images = np.append(self.images, image_list, 0)

        return image_list
