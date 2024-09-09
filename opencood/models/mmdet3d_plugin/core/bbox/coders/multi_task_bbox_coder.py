# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch

### for ce loss
import torch.nn.functional as F

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from opencood.models.mmdet3d_plugin.core.bbox.util import denormalize_bbox


@BBOX_CODERS.register_module()
class MultiTaskBBoxCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 use_sigmoid=True): ### for ce loss

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

        ### for ce loss with use_sigmoid
        self.use_sigmoid = use_sigmoid

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, task_ids):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        num_query = cls_scores.shape[0]
        '''
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]
        '''
        # exclude background
        if self.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
            scores, indexs = cls_scores.view(-1).topk(max_num)
            labels = indexs % self.num_classes
            bbox_index = indexs // self.num_classes
            task_index = torch.gather(task_ids, 1, labels.unsqueeze(1)).squeeze()
            bbox_preds = bbox_preds[task_index * num_query + bbox_index]
        else:
            scores, labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_num)
            labels = labels[bbox_index]
            task_index = torch.gather(task_ids, 1, labels.unsqueeze(1)).squeeze() ### TODO: 这个可能有些问题
            bbox_preds = bbox_preds[task_index * num_query + bbox_index]
        # cls_scores = cls_scores.sigmoid()
        # scores, indexs = cls_scores.view(-1).topk(max_num)
        # labels = indexs % self.num_classes
        # bbox_index = indexs // self.num_classes
        # task_index = torch.gather(task_ids, 1, labels.unsqueeze(1)).squeeze()
        #
        #
        # bbox_preds = bbox_preds[task_index * num_query + bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            query_index = bbox_index[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'query_index': query_index
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        task_num = len(preds_dicts)

        pred_bbox_list, pred_logits_list, task_ids_list = [], [], []
        for task_id in range(task_num):
            task_pred_dict = preds_dicts[task_id]
            ### remove the velocity
            task_pred_bbox = torch.cat(
                    (task_pred_dict['center'][-1], task_pred_dict['height'][-1],
                     task_pred_dict['dim'][-1], task_pred_dict['rot'][-1]),
                    dim=-1
                )
            task_pred_logits = task_pred_dict['cls_logits'][-1]
            pred_bbox_list.append(task_pred_bbox)
            pred_logits_list.append(task_pred_logits)

            task_ids = task_pred_logits.new_ones(task_pred_logits.shape).int() * task_id
            task_ids_list.append(task_ids)


        all_pred_logits = torch.cat(pred_logits_list, dim=-1)  # bs * nq * 10
        all_pred_bbox = torch.cat(pred_bbox_list, dim=1)  # bs * (task nq) * 10
        all_task_ids = torch.cat(task_ids_list, dim=-1) # bs * nq * 10

        batch_size = all_pred_logits.shape[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(all_pred_logits[i], all_pred_bbox[i], all_task_ids[i]))
        return predictions_list