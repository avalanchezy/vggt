# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat


from .blocks import EfficientUpdateFormer, CorrBlock
from .utils import sample_features4d, get_2d_embedding, get_2d_sincos_pos_embed
from .modules import Mlp


def to_batch_lists(
    values: Optional[Sequence], batch_size: int
) -> Optional[List[List[int]]]:
    """Convert possible tensor/list inputs into per-batch Python lists."""

    if values is None:
        return None

    if isinstance(values, torch.Tensor):
        if values.dim() == 1:
            return [values.cpu().tolist()]
        if values.dim() == 2:
            return [row.cpu().tolist() for row in values]
        return None

    if isinstance(values, (list, tuple)):
        if len(values) == batch_size and values and isinstance(values[0], (list, tuple, torch.Tensor)):
            batch_values: List[List[int]] = []
            for item in values:
                if isinstance(item, torch.Tensor):
                    batch_values.append(item.cpu().tolist())
                else:
                    batch_values.append(list(item))
            return batch_values
        if batch_size == 1 and values and isinstance(values[0], (int, float)):
            return [list(values)]

    return None


def parse_names_to_indices(
    frame_names: Sequence[Sequence[str]],
) -> Optional[Tuple[List[List[int]], List[List[int]]]]:
    """Parse timestamps and view indices from filename batches."""

    pattern = re.compile(r"^(?P<prefix>.+?)_(?P<time>\d+)_cam(?P<view>\d+)\.[^.]+$")
    time_batches: List[List[int]] = []
    view_batches: List[List[int]] = []

    for names in frame_names:
        current_times: List[int] = []
        current_views: List[int] = []
        for name in names:
            filename = os.path.basename(name)
            match = pattern.match(filename)
            if match is None:
                return None
            current_times.append(int(match.group("time")))
            current_views.append(int(match.group("view")))
        time_batches.append(current_times)
        view_batches.append(current_views)

    return time_batches, view_batches


def build_time_view_mask(times: Sequence[int], views: Sequence[int]) -> torch.Tensor:
    """Create a boolean attention mask enforcing adjacency constraints."""

    length = len(times)
    allowed = torch.zeros(length, length, dtype=torch.bool)

    time_to_view_idx: dict[int, dict[int, int]] = {}
    time_to_sorted_views: dict[int, List[int]] = {}

    for idx, (time_val, view_val) in enumerate(zip(times, views)):
        time_to_view_idx.setdefault(time_val, {})[view_val] = idx

    for time_val, view_map in time_to_view_idx.items():
        time_to_sorted_views[time_val] = sorted(view_map.keys())

    for idx, (time_val, view_val) in enumerate(zip(times, views)):
        neighbor_views: List[int] = []
        sorted_views = time_to_sorted_views[time_val]
        if not sorted_views:
            continue

        view_pos = sorted_views.index(view_val)
        num_views = len(sorted_views)
        neighbor_views.append(sorted_views[view_pos])
        if num_views > 1:
            neighbor_views.append(sorted_views[(view_pos - 1) % num_views])
            neighbor_views.append(sorted_views[(view_pos + 1) % num_views])

        for offset in (-1, 1):
            neighbor_time = time_val + offset
            candidate_map = time_to_view_idx.get(neighbor_time)
            if not candidate_map:
                continue
            for neighbor_view in neighbor_views:
                neighbor_idx = candidate_map.get(neighbor_view)
                if neighbor_idx is not None:
                    allowed[idx, neighbor_idx] = True
                    allowed[neighbor_idx, idx] = True

    if allowed.numel() > 0:
        row_has_edge = allowed.any(dim=1)
        missing_indices = torch.nonzero(~row_has_edge, as_tuple=True)[0]
        if missing_indices.numel() > 0:
            allowed[missing_indices, missing_indices] = True

    return ~allowed


def build_batch_time_view_mask(
    times: List[List[int]],
    views: List[List[int]],
    device: torch.device,
) -> torch.Tensor:
    """Stack per-sample masks into a tensor of shape (B, S, S)."""

    masks = [
        build_time_view_mask(sample_times, sample_views)
        for sample_times, sample_views in zip(times, views)
    ]
    return torch.stack(masks, dim=0).to(device)


class BaseTrackerPredictor(nn.Module):
    def __init__(
        self,
        stride=1,
        corr_levels=5,
        corr_radius=4,
        latent_dim=128,
        hidden_size=384,
        use_spaceatt=True,
        depth=6,
        max_scale=518,
        predict_conf=True,
    ):
        super(BaseTrackerPredictor, self).__init__()
        """
        The base template to create a track predictor
        
        Modified from https://github.com/facebookresearch/co-tracker/
        and https://github.com/facebookresearch/vggsfm
        """

        self.stride = stride
        self.latent_dim = latent_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.hidden_size = hidden_size
        self.max_scale = max_scale
        self.predict_conf = predict_conf

        self.flows_emb_dim = latent_dim // 2

        self.corr_mlp = Mlp(
            in_features=self.corr_levels * (self.corr_radius * 2 + 1) ** 2,
            hidden_features=self.hidden_size,
            out_features=self.latent_dim,
        )

        self.transformer_dim = self.latent_dim + self.latent_dim + self.latent_dim + 4

        self.query_ref_token = nn.Parameter(torch.randn(1, 2, self.transformer_dim))

        space_depth = depth if use_spaceatt else 0
        time_depth = depth

        self.updateformer = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.transformer_dim,
            hidden_size=self.hidden_size,
            output_dim=self.latent_dim + 2,
            mlp_ratio=4.0,
            add_space_attn=use_spaceatt,
        )

        self.fmap_norm = nn.LayerNorm(self.latent_dim)
        self.ffeat_norm = nn.GroupNorm(1, self.latent_dim)

        # A linear layer to update track feats at each iteration
        self.ffeat_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())

        self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

        if predict_conf:
            self.conf_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

    def forward(
        self,
        query_points,
        fmaps=None,
        iters=6,
        return_feat=False,
        down_ratio=1,
        apply_sigmoid=True,
        frame_time_indices=None,
        frame_view_indices=None,
        frame_names=None,
    ):
        """
        query_points: B x N x 2, the number of batches, tracks, and xy
        fmaps: B x S x C x HH x WW, the number of batches, frames, and feature dimension.
                note HH and WW is the size of feature maps instead of original images
        """
        B, N, D = query_points.shape
        B, S, C, HH, WW = fmaps.shape

        assert D == 2, "Input points must be 2D coordinates"

        # apply a layernorm to fmaps here
        fmaps = self.fmap_norm(fmaps.permute(0, 1, 3, 4, 2))
        fmaps = fmaps.permute(0, 1, 4, 2, 3)

        # Scale the input query_points because we may downsample the images
        # by down_ratio or self.stride
        # e.g., if a 3x1024x1024 image is processed to a 128x256x256 feature map
        # its query_points should be query_points/4
        if down_ratio > 1:
            query_points = query_points / float(down_ratio)

        query_points = query_points / float(self.stride)

        # Init with coords as the query points
        # It means the search will start from the position of query points at the reference frames
        coords = query_points.clone().reshape(B, 1, N, 2).repeat(1, S, 1, 1)

        # Sample/extract the features of the query points in the query frame
        query_track_feat = sample_features4d(fmaps[:, 0], coords[:, 0])

        # init track feats by query feats
        track_feats = query_track_feat.unsqueeze(1).repeat(1, S, 1, 1)  # B, S, N, C
        # back up the init coords
        coords_backup = coords.clone()

        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        time_lists = to_batch_lists(frame_time_indices, B)
        view_lists = to_batch_lists(frame_view_indices, B)

        if (time_lists is None or view_lists is None) and frame_names is not None:
            name_lists = frame_names
            if isinstance(name_lists, (list, tuple)) and name_lists and isinstance(name_lists[0], str):
                name_lists = [name_lists]
            parsed = parse_names_to_indices(name_lists) if name_lists is not None else None
            if parsed is not None:
                time_lists, view_lists = parsed

        time_attn_mask = None
        if time_lists is not None and view_lists is not None:
            time_attn_mask = build_batch_time_view_mask(time_lists, view_lists, device=query_points.device)

        coord_preds = []

        # Iterative Refinement
        for _ in range(iters):
            # Detach the gradients from the last iteration
            # (in my experience, not very important for performance)
            coords = coords.detach()

            fcorrs = fcorr_fn.corr_sample(track_feats, coords)

            corr_dim = fcorrs.shape[3]
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, corr_dim)
            fcorrs_ = self.corr_mlp(fcorrs_)

            # Movement of current coords relative to query points
            flows = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)

            flows_emb = get_2d_embedding(flows, self.flows_emb_dim, cat_coords=False)

            # (In my trials, it is also okay to just add the flows_emb instead of concat)
            flows_emb = torch.cat([flows_emb, flows / self.max_scale, flows / self.max_scale], dim=-1)

            track_feats_ = track_feats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            # Concatenate them as the input for the transformers
            transformer_input = torch.cat([flows_emb, fcorrs_, track_feats_], dim=2)

            # 2D positional embed
            # TODO: this can be much simplified
            pos_embed = get_2d_sincos_pos_embed(self.transformer_dim, grid_size=(HH, WW)).to(query_points.device)
            sampled_pos_emb = sample_features4d(pos_embed.expand(B, -1, -1, -1), coords[:, 0])

            sampled_pos_emb = rearrange(sampled_pos_emb, "b n c -> (b n) c").unsqueeze(1)

            x = transformer_input + sampled_pos_emb

            # Add the query ref token to the track feats
            query_ref_token = torch.cat(
                [self.query_ref_token[:, 0:1], self.query_ref_token[:, 1:2].expand(-1, S - 1, -1)], dim=1
            )
            x = x + query_ref_token.to(x.device).to(x.dtype)

            # B, N, S, C
            x = rearrange(x, "(b n) s d -> b n s d", b=B)

            # Compute the delta coordinates and delta track features
            delta, _ = self.updateformer(x, mask={"time": time_attn_mask} if time_attn_mask is not None else None)

            # BN, S, C
            delta = rearrange(delta, " b n s d -> (b n) s d", b=B)
            delta_coords_ = delta[:, :, :2]
            delta_feats_ = delta[:, :, 2:]

            track_feats_ = track_feats_.reshape(B * N * S, self.latent_dim)
            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)

            # Update the track features
            track_feats_ = self.ffeat_updater(self.ffeat_norm(delta_feats_)) + track_feats_

            track_feats = track_feats_.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)  # BxSxNxC

            # B x S x N x 2
            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)

            # Force coord0 as query
            # because we assume the query points should not be changed
            coords[:, 0] = coords_backup[:, 0]

            # The predicted tracks are in the original image scale
            if down_ratio > 1:
                coord_preds.append(coords * self.stride * down_ratio)
            else:
                coord_preds.append(coords * self.stride)

        # B, S, N
        vis_e = self.vis_predictor(track_feats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)
        if apply_sigmoid:
            vis_e = torch.sigmoid(vis_e)

        if self.predict_conf:
            conf_e = self.conf_predictor(track_feats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)
            if apply_sigmoid:
                conf_e = torch.sigmoid(conf_e)
        else:
            conf_e = None

        if return_feat:
            return coord_preds, vis_e, track_feats, query_track_feat, conf_e
        else:
            return coord_preds, vis_e, conf_e
