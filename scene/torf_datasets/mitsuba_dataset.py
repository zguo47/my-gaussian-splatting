import numpy as np

from scene.torf_datasets.tof_dataset import *

from scene.torf_datasets.my_utils.projection_utils import *


class MitsubaDataset(ToFDataset):
    def __init__(
        self,
        args,
        file_endings={
            "tof": "npy",
            "color": "npy",
            "depth": "npy",
            "motion_mask": "npy",
            "cams": "npy",
        },
    ):
        super().__init__(args, file_endings)

    def _read_tof(self, tof_filename):
        return np.load(tof_filename)

    def _read_color(self, color_filename):
        return np.load(color_filename)

    def _read_depth(self, depth_filename):
        return np.load(depth_filename)

    def _read_motion_mask(self, motion_mask_filename):
        return np.load(motion_mask_filename)

    def _process_camera_params(self, args):
        super()._process_camera_params(args)
