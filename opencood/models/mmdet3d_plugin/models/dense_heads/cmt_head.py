# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import math
import copy
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.core import multi_apply
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid


def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


@HEADS.register_module()
class CmtHead(BaseModule):

    def __init__(self,
                 in_channels,
                 num_query=900,
                 hidden_dim=128,
                 depth_num=64,
                 norm_bbox=True,
                 downsample_scale=8,
                 split=0.75,
                 train_cfg=None,
                 test_cfg=None,
                 transformer=None,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None
        super(CmtHead, self).__init__(init_cfg=init_cfg)

        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.in_channels = in_channels
        self.depth_num = depth_num
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale

        self.split = split

        self.pc_range = self.train_cfg.point_cloud_range

        self.shared_conv = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d")
        )

        # transformer
        self.transformer = build_transformer(transformer)
        self.reference_points = nn.Embedding(num_query, 3)
        self.bev_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rv_embedding = nn.Sequential(
            nn.Linear(self.depth_num * 3, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )

    def init_weights(self):
        super(CmtHead, self).init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    # def check_tensor_for_invalid_values(self, tensor):
    #     """
    #     检查PyTorch张量是否包含无效值：nan, inf, -inf。
    #
    #     参数:
    #     tensor -- 要检查的PyTorch张量
    #
    #     返回:
    #     has_nan -- 张量中是否有nan值
    #     has_inf -- 张量中是否有inf值
    #     has_neg_inf -- 张量中是否有-inf值
    #     """
    #     # 检查是否有任何 nan 值
    #     has_nan = torch.isnan(tensor).any().item()
    #
    #     # 检查是否有任何正无穷大或负无穷大值
    #     has_inf = torch.isinf(tensor).any().item()
    #
    #     # 检查是否有任何负无穷大值
    #     has_neg_inf = (tensor == float('-inf')).any().item()
    #
    #     return has_nan, has_inf, has_neg_inf

    @property
    def coords_bev(self):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,
            cfg['grid_size'][0] // self.downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0)  # (H*W, 2)
        return coord_base

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _rv_pe(self, img_feats, img_metas):
        BN, C, H, W = img_feats.shape
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W
        coords_d = 1 + torch.arange(self.depth_num, device=img_feats[0].device).float() * (
                    self.pc_range[3] - 1) / self.depth_num
        coords_h, coords_w, coords_d = torch.meshgrid([coords_h, coords_w, coords_d])

        coords = torch.stack([coords_w, coords_h, coords_d, coords_h.new_ones(coords_h.shape)], dim=-1)
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]

        imgs2lidars = np.concatenate([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)
        coords_3d = torch.einsum('hwdo, bco -> bhwdc', coords, imgs2lidars)
        coords_3d = (coords_3d[..., :3] - coords_3d.new_tensor(self.pc_range[:3])[None, None, None, :]) \
                    / (coords_3d.new_tensor(self.pc_range[3:]) - coords_3d.new_tensor(self.pc_range[:3]))[None, None,
                      None, :]
        ### if use FSDP or DeepSpeed, please open the blow code.
        # coords_3d = coords_3d.to(img_feats)
        return self.rv_embedding(coords_3d.reshape(*coords_3d.shape[:-2], -1))

    def _bev_query_embed(self, ref_points, img_metas):
        bev_embeds = self.bev_embedding(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return bev_embeds

    def _rv_query_embed(self, ref_points, img_metas):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(lidars2imgs).float().to(ref_points.device)
        imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(ref_points.device)

        ref_points = ref_points * (ref_points.new_tensor(self.pc_range[3:]) - ref_points.new_tensor(
            self.pc_range[:3])) + ref_points.new_tensor(self.pc_range[:3])
        proj_points = torch.einsum('bnd, bvcd -> bvnc',
                                   torch.cat([ref_points, ref_points.new_ones(*ref_points.shape[:-1], 1)], dim=-1),
                                   lidars2imgs)

        proj_points_clone = proj_points.clone()
        z_mask = proj_points_clone[..., 2:3].detach() > 0
        proj_points_clone[..., :3] = proj_points[..., :3] / (
                    proj_points[..., 2:3].detach() + z_mask * 1e-6 - (~z_mask) * 1e-6)

        mask = (proj_points_clone[..., 0] < pad_w) & (proj_points_clone[..., 0] >= 0) & (
                    proj_points_clone[..., 1] < pad_h) & (proj_points_clone[..., 1] >= 0)
        mask &= z_mask.squeeze(-1)

        coords_d = 1 + torch.arange(self.depth_num, device=ref_points.device).float() * (
                    self.pc_range[3] - 1) / self.depth_num
        proj_points_clone = torch.einsum('bvnc, d -> bvndc', proj_points_clone, coords_d)
        proj_points_clone = torch.cat(
            [proj_points_clone[..., :3], proj_points_clone.new_ones(*proj_points_clone.shape[:-1], 1)], dim=-1)
        projback_points = torch.einsum('bvndo, bvco -> bvndc', proj_points_clone, imgs2lidars)

        projback_points = (projback_points[..., :3] - projback_points.new_tensor(self.pc_range[:3])[None, None, None,
                                                      :]) \
                          / (projback_points.new_tensor(self.pc_range[3:]) - projback_points.new_tensor(
            self.pc_range[:3]))[None, None, None, :]

        rv_embeds = self.rv_embedding(projback_points.reshape(*projback_points.shape[:-2], -1))
        rv_embeds = (rv_embeds * mask.unsqueeze(-1)).sum(dim=1)
        return rv_embeds

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        bev_embeds = self._bev_query_embed(ref_points, img_metas)
        rv_embeds = self._rv_query_embed(ref_points, img_metas)
        return bev_embeds, rv_embeds

    def forward_single(self, x, x_img, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        ret_dicts = []
        x = self.shared_conv(x)

        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)

        rv_pos_embeds = self._rv_pe(x_img, img_metas)
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))

        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
        query_embeds = bev_query_embeds + rv_query_embeds

        outs_dec, _ = self.transformer(
            x, x_img, query_embeds,
            bev_pos_embeds, rv_pos_embeds,
            attn_masks=attn_mask
        )
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())

        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height

            ret_dicts.append(outs)

        return ret_dicts

    def forward(self, pts_feats, img_feats=None, img_metas=None, fuse_metas=None):
        """
            list([bs, c, h, w])
        """
        img_metas = [img_metas for _ in range(len(pts_feats))]
        return multi_apply(self.forward_single, pts_feats, img_feats, img_metas, fuse_metas=fuse_metas)


@HEADS.register_module()
class CmtImageHead(CmtHead):

    def __init__(self, *args, **kwargs):
        super(CmtImageHead, self).__init__(*args, **kwargs)
        self.shared_conv = None

    def forward_single(self, x, x_img, img_metas, fuse_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        assert x is None
        with torch.cuda.amp.autocast(enabled=False):
            reference_points = self.reference_points.weight
            reference_points, attn_mask, mask_dict = self.prepare_for_dn(len(img_metas), reference_points, img_metas)

            rv_pos_embeds = self._rv_pe(x_img, img_metas)

            bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
            query_embeds = bev_query_embeds + rv_query_embeds

        outs_dec, _ = self.transformer(
            x_img, query_embeds,
            rv_pos_embeds,
            attn_masks=attn_mask,
            bs=len(img_metas)
        )
        outs_dec = torch.nan_to_num(outs_dec)

        outs_dict = dict()
        outs_dict['outs_dec'] = outs_dec
        outs_dict['reference_points'] = reference_points

        return [outs_dict]


@HEADS.register_module()
class CmtLidarHead(CmtHead):

    def __init__(self, *args, **kwargs):
        super(CmtLidarHead, self).__init__(*args, **kwargs)
        self.rv_embedding = None

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        bev_embeds = self._bev_query_embed(ref_points, img_metas)
        return bev_embeds, None

    def forward_single(self, x, x_img, img_metas, fuse_metas=None):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        assert x_img is None
        x = self.shared_conv(x)
        with torch.cuda.amp.autocast(enabled=False):
            reference_points = self.reference_points.weight
            reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)

            mask = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])

            bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
            bev_query_embeds, _ = self.query_embed(reference_points, img_metas)

            query_embeds = bev_query_embeds

        outs_dec, _ = self.transformer(
            x, mask, query_embeds,
            bev_pos_embeds,
            attn_masks=attn_mask
        )
        outs_dec = torch.nan_to_num(outs_dec)

        outs_dict = dict()
        outs_dict['outs_dec'] = outs_dec
        outs_dict['reference_points'] = reference_points

        return [outs_dict]