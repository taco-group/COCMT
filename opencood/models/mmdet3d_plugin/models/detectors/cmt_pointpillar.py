# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
import numpy as np

from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from opencood.models.mmdet3d_plugin.models.utils.grid_mask import GridMask

### pointpillar
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv

@DETECTORS.register_module()
class CmtDetector_PointPillar(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 lidar_backbone=None,
                 **kwargs):
        super(CmtDetector_PointPillar, self).__init__(**kwargs)

        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        if lidar_backbone is not None:
            ### pointpillar
            # PIllar VFE
            self.pillar_vfe = PillarVFE(lidar_backbone['pillar_vfe'],
                                        num_point_features=4,
                                        voxel_size=np.array(kwargs['train_cfg']['pts']['voxel_size']),
                                        point_cloud_range=np.array(kwargs['train_cfg']['pts']['point_cloud_range']))
            self.scatter = PointPillarScatter(lidar_backbone['point_pillar_scatter'])
            if lidar_backbone.get('resnet_bev_backbone', False):
                self.backbone = ResNetBEVBackbone(lidar_backbone['resnet_bev_backbone'])
            elif lidar_backbone.get('base_bev_backbone', False):
                self.backbone = BaseBEVBackbone(lidar_backbone['base_bev_backbone'], 64)
            else:
                raise NotImplementedError('only base_bev_backbone and resnet_bev_backbone is supported')
            # used to downsample the feature map for efficient computation
            self.shrink_conv = DownsampleConv(lidar_backbone['shrink_header'])

    def init_weights(self):
        """Initialize model weights."""
        super(CmtDetector_PointPillar, self).init_weights()

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None

        batch_dict = self.pillar_vfe(pts)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        x = self.shrink_conv(spatial_features_2d)

        return [x]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      proposals=None,
                      fuse_metas=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)  ### pts_feats (N, C, H, W)
        if pts_feats or img_feats:
            outs = self.forward_pts_train(pts_feats, img_feats,
                                          img_metas, fuse_metas)
        return outs

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          img_metas,
                          fuse_metas=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas, fuse_metas)
        return outs