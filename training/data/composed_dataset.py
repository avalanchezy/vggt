# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
import os
import re

from hydra.utils import instantiate
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import bisect
from .dataset_util import *
from .track_util import *
from .augmentation import get_image_augmentation


def _parse_time_view_order(image_paths):
    """Parse frame indices from filenames and return sorting order and metadata.

    Filenames are expected to follow the pattern ``<prefix>_XXXXXX_camY.ext`` where
    ``XXXXXX`` is a zero-padded timestamp and ``Y`` is the camera/view index. The
    function sorts frames by ascending timestamp and then by ascending view index
    within each timestamp. Missing views are simply ignored.

    Args:
        image_paths (list[str]): File paths for the frames in the current sample.

    Returns:
        tuple[list[int] | None, list[int] | None, list[int] | None]:
            The permutation order to achieve the desired sorting, along with the
            sorted time indices and view indices. ``None`` values are returned when
            the paths are unavailable or cannot be parsed.
    """

    if not image_paths:
        return None, None, None

    pattern = re.compile(r"^(?P<prefix>.+?)_(?P<time>\d+)_cam(?P<view>\d+)\.[^.]+$")
    parsed = []

    for idx, path in enumerate(image_paths):
        filename = os.path.basename(path)
        match = pattern.match(filename)
        if match is None:
            return None, None, None
        time_idx = int(match.group("time"))
        view_idx = int(match.group("view"))
        parsed.append((time_idx, view_idx, idx))

    parsed.sort(key=lambda item: (item[0], item[1]))

    order = [item[2] for item in parsed]
    time_indices = [item[0] for item in parsed]
    view_indices = [item[1] for item in parsed]
    return order, time_indices, view_indices


def _reorder_sequence_items(batch_dict, order):
    """Apply a permutation along the sequence dimension for supported keys."""

    if order is None:
        return

    sequence_keys = [
        "images",
        "depths",
        "extrinsics",
        "intrinsics",
        "cam_points",
        "world_points",
        "point_masks",
        "image_paths",
        "tracks",
        "track_masks",
    ]

    for key in sequence_keys:
        if key not in batch_dict or batch_dict[key] is None:
            continue
        seq_value = batch_dict[key]
        if isinstance(seq_value, list):
            batch_dict[key] = [seq_value[i] for i in order]
        else:
            batch_dict[key] = seq_value[order]

    if "ids" in batch_dict and batch_dict["ids"] is not None:
        batch_dict["ids"] = batch_dict["ids"][order]


class ComposedDataset(Dataset, ABC):
    """
    Composes multiple base datasets and applies common configurations.

    This dataset provides a flexible way to combine multiple base datasets while
    applying shared augmentations, track generation, and other processing steps.
    It handles image normalization, tensor conversion, and other preparations
    needed for training computer vision models with sequences of images.
    """
    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        """
        Initializes the ComposedDataset.

        Args:
            dataset_configs (dict): List of Hydra configurations for base datasets.
            common_config (dict): Shared configurations (augs, tracks, mode, etc.).
            **kwargs: Additional arguments (unused).
        """
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use custom concatenation class that supports tuple indexing
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        # --- Augmentation Settings ---
        # Controls whether to apply identical color jittering across all frames in a sequence
        self.cojitter = common_config.augs.cojitter
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = common_config.augs.cojitter_ratio
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = get_image_augmentation(
            color_jitter=common_config.augs.color_jitter,
            gray_scale=common_config.augs.gray_scale,
            gau_blur=common_config.augs.gau_blur,
        )

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_config.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        # --- Track Settings ---
        # Whether to include point tracks in the output
        self.load_track = common_config.load_track
        # Number of point tracks to include per sequence
        self.track_num = common_config.track_num

        # --- Mode Settings ---
        # Whether the dataset is being used for training (affects augmentations)
        self.training = common_config.training
        self.common_config = common_config

        self.total_samples = len(self.base_dataset)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples


    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts to PyTorch tensors, applies augmentations,
        and prepares tracks if enabled.

        Args:
            idx_tuple (tuple): a tuple of (seq_idx, num_images, aspect_ratio)

        Returns:
            dict: A dictionary containing the sequence data (images, poses, tracks, etc.).
        """
        # If fixed settings are provided, override the tuple values
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]
        seq_name = batch["seq_name"]

        order = None
        time_indices = None
        view_indices = None
        if "image_paths" in batch:
            order, time_indices, view_indices = _parse_time_view_order(batch["image_paths"])
            _reorder_sequence_items(batch, order)

        # --- Data Conversion and Preparation ---
        # Convert numpy arrays to tensors
        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        # Normalize images from [0, 255] to [0, 1]
        images = images.permute(0,3,1,2).to(torch.get_default_dtype()).div(255)

        # Convert other data to tensors with appropriate types
        depths = torch.from_numpy(np.stack(batch["depths"]).astype(np.float32))
        extrinsics = torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32))
        cam_points = torch.from_numpy(np.stack(batch["cam_points"]).astype(np.float32))
        world_points = torch.from_numpy(np.stack(batch["world_points"]).astype(np.float32))
        point_masks = torch.from_numpy(np.stack(batch["point_masks"])) # Mask indicating valid depths / world points / cam points per frame
        ids = torch.from_numpy(batch["ids"])    # Frame indices sampled from the original sequence


        # --- Apply Color Augmentation (training mode only) ---
        if self.training and self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                # Apply the same color jittering transformation to all frames
                images = self.image_aug(images)
            else:
                # Apply different color jittering to each frame individually
                for aug_img_idx in range(len(images)):
                    images[aug_img_idx] = self.image_aug(images[aug_img_idx])


        # --- Prepare Final Sample Dictionary ---
        sample = {
            "seq_name": seq_name,
            "ids": ids,
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
        }

        if time_indices is not None and view_indices is not None:
            sample["frame_time_indices"] = torch.as_tensor(time_indices, dtype=torch.long)
            sample["frame_view_indices"] = torch.as_tensor(view_indices, dtype=torch.long)

        if "image_paths" in batch:
            sample["image_paths"] = batch["image_paths"]

        # --- Track Processing (if enabled) ---
        if self.load_track:
            if batch["tracks"] is not None:
                # Use pre-computed tracks from the dataset
                tracks = torch.from_numpy(np.stack(batch["tracks"]).astype(np.float32))
                track_vis_mask = torch.from_numpy(np.stack(batch["track_masks"]).astype(bool))

                # Sample a subset of tracks randomly
                valid_indices = torch.where(track_vis_mask[0])[0]
                if len(valid_indices) >= self.track_num:
                    # If we have enough tracks, sample without replacement
                    sampled_indices = valid_indices[torch.randperm(len(valid_indices))][:self.track_num]
                else:
                    # If not enough tracks, sample with replacement (allow duplicates)
                    sampled_indices = valid_indices[torch.randint(0, len(valid_indices),
                                                    (self.track_num,),
                                                    dtype=torch.int64,
                                                    device=valid_indices.device)]

                # Extract the sampled tracks and their masks
                tracks = tracks[:, sampled_indices, :]
                track_vis_mask = track_vis_mask[:, sampled_indices]
                track_positive_mask = torch.ones(track_vis_mask.shape[1]).bool()

            else:
                # Generate tracks on-the-fly using depth information
                # This creates synthetic tracks based on the 3D information available
                tracks, track_vis_mask, track_positive_mask = build_tracks_by_depth(
                    extrinsics, intrinsics, world_points, depths, point_masks, images,
                    target_track_num=self.track_num, seq_name=seq_name
                )

            # Add track information to the sample dictionary
            sample["tracks"] = tracks
            sample["track_vis_mask"] = track_vis_mask
            sample["track_positive_mask"] = track_positive_mask

        return sample


class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """
    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = common_config.inside_random

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]
