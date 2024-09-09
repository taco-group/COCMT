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

class CMTCamera(nn.Module):
    def __init__(self, config):
        super(CMTCamera, self).__init__()

        self.pc_range = config['pc_range']
        cfg = config['CMT_C']['cfg']
        self.cfg = cfg
        self.config = config
        self.cmt_c = build_detector(cfg.model)
        self.cmt_c.init_weights()

        if 'Camera_SeperateHead' in config:

            camera_separate_head_cfg = config['Camera_SeperateHead']
            common_heads = camera_separate_head_cfg['common_heads']
            num_cls = camera_separate_head_cfg['num_cls']
            heads = copy.deepcopy(common_heads)
            heads.update(dict(cls_logits=[num_cls, 2]))
            camera_separate_head_cfg.update(heads=heads)

            self.camera_task_heads = nn.ModuleList()
            self.camera_task_heads.append(builder.build_head(camera_separate_head_cfg))
            self.camera_task_heads[0].init_weights()

        self.img_shape = config['CMT_C']['img_shape']
        self.resize_ratio = config['CMT_C'].get('resize_ratio', False)

        self.return_features = False

    def reformat_input(self, batch):
        # (B,N,3,H,W) H600 W800
        img = batch['inputs_m2']['imgs'].contiguous() ### change the img's format to nuscenes
        bs = img.shape[0]
        num_cam = img.shape[1]

        cav2cam = batch['inputs_m2']['lidar2cameras']

        intrinsics = batch['inputs_m2']['intrins']
        if self.resize_ratio:
            scale_h, scale_w = self.resize_ratio
            adjusted_intrinsics = intrinsics.clone()
            adjusted_intrinsics[:, :, 0, 0] *= scale_w  # 调整 fx
            adjusted_intrinsics[:, :, 1, 1] *= scale_h  # 调整 fy
            adjusted_intrinsics[:, :, 0, 2] *= scale_w  # 调整 cx
            adjusted_intrinsics[:, :, 1, 2] *= scale_h  # 调整 cy
            intrinsics = adjusted_intrinsics

        intrinsics_hom = torch.eye(4).reshape((1, 1, 4, 4)).repeat(
            (bs, num_cam, 1, 1)).to(intrinsics.device)
        intrinsics_hom[:, :, :3, :3] = intrinsics
        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):
        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y
        # (x, y ,z) -> (y, -z, x)
        flip_matrix = torch.Tensor(
            [[0, 1, 0, 0],
             [0, 0, -1, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1]]).reshape(
            (1, 1, 4, 4)).repeat(
            (bs, num_cam, 1, 1)).to(intrinsics.device)
        flip_matrix[..., 1, 1] = -1

        cav2cam = cav2cam.to(torch.float32)

        lidar2img = torch.matmul(flip_matrix, cav2cam)
        lidar2img = torch.matmul(intrinsics_hom,
                                 lidar2img).detach().to(torch.float32).cpu().numpy()
        img_metas = []
        for i in range(bs):
            img_meta = {}
            img_meta['img_shape'] = [(self.img_shape[0], self.img_shape[1]) for
                                     _ in range(num_cam)]
            ### 本文没有实现 padding 操作,为了 mmdet3d 对齐,直接去 image_shape,加上通道维度.
            img_meta['pad_shape'] = [(self.img_shape[0], self.img_shape[1], int(3)) for
                                     _ in range(num_cam)]
            img_meta['lidar2img'] = []
            for j in range(num_cam):
                img_meta['lidar2img'].append(lidar2img[i, j])
            img_metas.append(img_meta)

        return img, img_metas

    def set_return_features(self):
        self.return_features = True

    def forward(self, batch, test_flag=False):

        img, img_metas = self.reformat_input(batch)

        outs = self.cmt_c.forward_train(points=None, img_metas=img_metas, img=img)

        # DL, B, Q, ED -> Decoder Layers, Batch size, Num Queries, Embedded Dim
        outs_dec = outs[0][0]['outs_dec']
        # B, Q, 3 -> Batch size, Num Queries, XYZ
        reference_points = outs[0][0]['reference_points']

        with torch.cuda.amp.autocast(enabled=False):

            reference = inverse_sigmoid(reference_points.clone())

            ret_dicts = []

            for task_id, task in enumerate(self.camera_task_heads, 0):
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
                bbox_list = self.camera_task_heads[0].get_bboxes(
                    ret_dicts, img_metas=None, rescale=False)
                bbox_results = [
                    bbox3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels, _ in bbox_list
                ]
                return bbox_results[0]