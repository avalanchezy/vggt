# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.heads.track_modules.base_track_predictor import (
    parse_names_to_indices,
    to_batch_lists,
)


def _normalize_name_batches(
    frame_names: Optional[Union[Sequence[str], Sequence[Sequence[str]]]],
    batch_size: int,
) -> Optional[List[List[str]]]:
    if frame_names is None:
        return None

    if isinstance(frame_names, Sequence) and not isinstance(frame_names, (str, bytes)) and frame_names and isinstance(frame_names[0], str):
        if batch_size == 1:
            return [list(frame_names)]
        return None

    if (
        isinstance(frame_names, Sequence)
        and not isinstance(frame_names, (str, bytes))
        and frame_names
        and isinstance(frame_names[0], Sequence)
    ):
        normalized: List[List[str]] = []
        for names in frame_names:  # type: ignore[assignment]
            normalized.append([str(name) for name in names])
        if len(normalized) == batch_size:
            return normalized
    return None


def _sort_sequences_by_time_view(
    images: torch.Tensor,
    frame_time_indices: Optional[Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]]],
    frame_view_indices: Optional[Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]]],
    frame_names: Optional[Union[Sequence[str], Sequence[Sequence[str]]]],
) -> Tuple[
    torch.Tensor,
    Optional[List[List[int]]],
    Optional[List[List[int]]],
    Optional[List[List[str]]],
]:
    batch_size, seq_len = images.shape[0], images.shape[1]

    time_lists = to_batch_lists(frame_time_indices, batch_size)
    view_lists = to_batch_lists(frame_view_indices, batch_size)
    name_lists = _normalize_name_batches(frame_names, batch_size)

    if (time_lists is None or view_lists is None) and name_lists is not None:
        parsed = parse_names_to_indices(name_lists)
        if parsed is not None:
            time_lists, view_lists = parsed

    if time_lists is None or view_lists is None:
        return images, None, None, name_lists

    sorted_images: List[torch.Tensor] = []
    sorted_time_lists: List[List[int]] = []
    sorted_view_lists: List[List[int]] = []
    sorted_name_lists: Optional[List[List[str]]] = [] if name_lists is not None else None
    metadata_valid = True

    for batch_idx in range(batch_size):
        times = time_lists[batch_idx]
        views = view_lists[batch_idx]
        names = name_lists[batch_idx] if name_lists is not None else None

        if len(times) != seq_len or len(views) != seq_len:
            metadata_valid = False
            sorted_images.append(images[batch_idx])
            sorted_time_lists.append([int(value) for value in times])
            sorted_view_lists.append([int(value) for value in views])
            if sorted_name_lists is not None:
                sorted_name_lists.append(list(names) if names is not None else [])
            continue

        order = sorted(range(seq_len), key=lambda idx: (times[idx], views[idx]))
        order_tensor = torch.tensor(order, dtype=torch.long, device=images.device)

        sorted_images.append(images[batch_idx].index_select(0, order_tensor))
        sorted_time_lists.append([int(times[idx]) for idx in order])
        sorted_view_lists.append([int(views[idx]) for idx in order])
        if sorted_name_lists is not None:
            if names is not None:
                sorted_name_lists.append([names[idx] for idx in order])
            else:
                sorted_name_lists.append([])

    images_sorted = torch.stack(sorted_images, dim=0)

    if not metadata_valid:
        return images_sorted, None, None, sorted_name_lists

    return images_sorted, sorted_time_lists, sorted_view_lists, sorted_name_lists


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(
        self,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        frame_time_indices: Optional[torch.Tensor] = None,
        frame_view_indices: Optional[torch.Tensor] = None,
        frame_names: Optional[Union[Sequence[str], Sequence[Sequence[str]]]] = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            if frame_time_indices is not None and isinstance(frame_time_indices, torch.Tensor) and frame_time_indices.dim() == 1:
                frame_time_indices = frame_time_indices.unsqueeze(0)
            if frame_view_indices is not None and isinstance(frame_view_indices, torch.Tensor) and frame_view_indices.dim() == 1:
                frame_view_indices = frame_view_indices.unsqueeze(0)
            if frame_names is not None and isinstance(frame_names, (list, tuple)) and frame_names and isinstance(frame_names[0], str):
                frame_names = [list(frame_names)]

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        (
            images,
            frame_time_lists,
            frame_view_lists,
            frame_name_lists,
        ) = _sort_sequences_by_time_view(images, frame_time_indices, frame_view_indices, frame_names)

        frame_time_indices_for_tracker = frame_time_lists
        frame_view_indices_for_tracker = frame_view_lists
        frame_names_for_tracker = frame_name_lists
        if frame_names_for_tracker is None and frame_names is not None:
            if (
                isinstance(frame_names, Sequence)
                and not isinstance(frame_names, (str, bytes))
                and len(frame_names) > 0
                and isinstance(frame_names[0], str)
            ):
                frame_names_for_tracker = [list(frame_names)]
            elif (
                isinstance(frame_names, Sequence)
                and not isinstance(frame_names, (str, bytes))
                and len(frame_names) > 0
                and isinstance(frame_names[0], Sequence)
            ):
                frame_names_for_tracker = [list(names) for names in frame_names]
            else:
                frame_names_for_tracker = frame_names

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
                frame_time_indices=frame_time_indices_for_tracker,
                frame_view_indices=frame_view_indices_for_tracker,
                frame_names=frame_names_for_tracker,
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

