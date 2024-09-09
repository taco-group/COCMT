### add the CMT-L models main function

import torch
import torch.nn as nn

from mmdet3d.models import build_detector
from mmdet3d.models import builder
from mmdet3d.core import bbox3d2result
from opencood.models.mmdet3d_plugin import *
import copy

### sigmoid
from mmdet.models.utils.transformer import inverse_sigmoid

### topk
from opencood.models.mmdet3d_plugin.models.utils.mln_utils import topk_gather
import torch.nn.functional as F

class CMTLiDAR(nn.Module):
    def __init__(self, config):
        super(CMTLiDAR, self).__init__()

        self.pc_range = config['pc_range']
        cfg = config['CMT_L']['cfg']
        self.cfg = cfg
        self.config = config
        self.cmt_l = build_detector(cfg.model)
        self.cmt_l.init_weights()

        if 'Lidar_SeperateHead' in config:

            lidar_separate_head_cfg = config['Lidar_SeperateHead']
            common_heads = lidar_separate_head_cfg['common_heads']
            num_cls = lidar_separate_head_cfg['num_cls']
            heads = copy.deepcopy(common_heads)
            heads.update(dict(cls_logits=[num_cls, 2]))
            lidar_separate_head_cfg.update(heads=heads)

            self.lidar_task_heads = nn.ModuleList()
            self.lidar_task_heads.append(builder.build_head(lidar_separate_head_cfg))
            self.lidar_task_heads[0].init_weights()


        self.return_features = False

    def set_return_features(self):
        self.return_features = True

    def forward(self, batch, test_flag=False):

        voxel_features = batch['inputs_m1']['voxel_features']
        voxel_coords = batch['inputs_m1']['voxel_coords']
        voxel_num_points = batch['inputs_m1']['voxel_num_points']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'batch_size': voxel_coords[-1, 0] + 1}

        outs = self.cmt_l.forward_train(points=batch_dict)

        outs_dec = outs[0][0]['outs_dec']

        reference_points = outs[0][0]['reference_points']

        with torch.cuda.amp.autocast(enabled=False):

            reference = inverse_sigmoid(reference_points.clone())

            ret_dicts = []

            for task_id, task in enumerate(self.lidar_task_heads, 0):
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

            if self.return_features:
                ### build 3d object centers
                object_center = outs['center']
                object_height = outs['height']
                object_centers = torch.cat((object_center, object_height), dim=3)

                # TODO: put the k into the config(yaml)
                ### build topk indexes
                cls_logits = outs['cls_logits'][-1]
                score = F.softmax(cls_logits, dim=-1)[..., 0:1] ### only for object, not for background ### because of the celoss, we use softmax instead of sigmoid here
                object_scores, topk_indexes = torch.topk(score, 120, dim=1)

                # TODO: does is need to be detached here?
                ### topk proposals
                object_query = topk_gather(outs_dec[-1], topk_indexes)
                reference_points = topk_gather(reference_points, topk_indexes)
                object_centers = topk_gather(object_centers[-1], topk_indexes)

                return ret_dicts, object_query, reference_points, object_centers, object_scores

            all_ret_dicts = {}
            all_ret_dicts['single_ego_ret_dicts'] = ret_dicts

            if test_flag is not True:
                # train
                return all_ret_dicts
            else:
                # test
                bbox_list = self.lidar_task_heads[0].get_bboxes(
                    ret_dicts, img_metas=None, rescale=False)
                bbox_results = [
                    bbox3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels, _ in bbox_list
                ]
                return bbox_results[0]