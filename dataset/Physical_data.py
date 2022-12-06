import bisect
from functools import partial
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from l5kit.data import ChunkedDataset, get_frames_slice_from_scenes
from l5kit.dataset.utils import convert_str_to_fixed_length_tensor
from l5kit.vectorization.vectorizer import Vectorizer

from Generate_Vectorized import generate_physical_vectorized


class PhysicalDataset(Dataset):
    def __init__(self, cfg, zarr_dataset, vectorizer):
        """
        Get a PyTorch dataset including physical property

        Args:
            cfg : configuration file
            zarr_dataset : the raw zarr dataset
            vectorizer : a object that supports vectorization around an AV
        """
        self.cfg = cfg
        self.dataset = zarr_dataset
        self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]
        self.vectorizer = vectorizer

        # build a partial so we don't have to access cfg each time
        self.sample_function = self._get_sample_function()

    def _get_sample_function(self) :
        return partial(
            generate_physical_vectorized,
            history_num_frames=self.cfg["model_params"]["history_num_frames"],
            future_num_frames=self.cfg["model_params"]["future_num_frames"],
            filter_agents_threshold=self.cfg["raster_params"]["filter_agents_threshold"],
            vectorizer=self.vectorizer,
            max_num_agent = self.cfg["data_generation_params"]["max_num_agent"],
            max_num_lanes = self.cfg["data_generation_params"]["max_num_lanes"],
        )

    def __len__(self) :
        """
        Get the number of available AV frames

        Returns:
            the number of elements in the dataset
        """
        return len(self.dataset.frames)

    def get_frame(self, scene_index, state_index, track_id=None) :
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame

        Args:
            scene_index : the index of the scene in the zarr
            state_index : a relative frame index in the scene
            track_id : the agent to rasterize or None for the AV
        Returns:
            data : dict that contains physical property

        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]

        tl_faces = self.dataset.tl_faces

        data = self.sample_function(state_index, frames, self.dataset.agents, tl_faces, track_id)

        data["scene_index"] = scene_index
        data["host_id"] = np.uint8(convert_str_to_fixed_length_tensor(self.dataset.scenes[scene_index]["host"]).cpu())
        data["timestamp"] = frames[state_index]["timestamp"]
        data["track_id"] = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        return data

    def __getitem__(self, index) :
        """
        Function called by Torch to get an element

        Args:
            index (int): index of the element to retrieve

        Returns: please look get_frame signature and docstring

        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        scene_index = bisect.bisect_right(self.cumulative_sizes, index)

        if scene_index == 0:
            state_index = index
        else:
            state_index = index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index)

'''
    def get_scene_dataset(self, scene_index: int) :
        """
        Returns another PhysicalDataset dataset where the underlying data can be modified.

        Args:
            scene_index : the scene index of the new dataset

        Returns:
            PhysicalDataset: A valid dataset with a copy of the data

        """
        dataset = self.dataset.get_scene_dataset(scene_index)
        return PhysicalDataset(self.cfg, dataset, self.vectorizer)
'''