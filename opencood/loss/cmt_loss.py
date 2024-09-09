import torch
import torch.nn as nn

import math
import copy
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply)
from mmdet.models import build_loss

import collections
from opencood.models.mmdet3d_plugin.core.bbox.util import normalize_bbox

class CMTLoss(nn.Module):
    def __init__(self, args,):
        super(CMTLoss, self).__init__()
        loss_cls = args['loss_cls']
        loss_bbox = args['loss_bbox']
        tasks = [args['tasks']]
        assigner = args['assigner']
        self.pc_range = args['pc_range']
        self.code_weights = args['code_weights']


        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]

        ### for ce loss DETR
        self.bg_cls_weight = 0
        # self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None:
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(self.num_classes[0] + 1) * class_weight
            # set background class as the last indice
            class_weight[self.num_classes[0]] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        ### build loss
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        ### build assigner and sampler
        self.assigner = build_assigner(assigner)
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)

        self.loss_dict = dict()

    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:

            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (list[Tensor]): num_tasks x (num_query, 10)
            pred_logits (list[Tensor]): num_tasks x (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (list[Tensor]): num_tasks x (num_query, ).
                - label_weights_tasks (list[Tensor]): num_tasks x (num_query, ).
                - bbox_targets_tasks (list[Tensor]): num_tasks x (num_query, 9).
                - bbox_weights_tasks (list[Tensor]): num_tasks x (num_query, 10).
                - pos_inds (list[Tensor]): num_tasks x Sampled positive indices.
                - neg_inds (Tensor): num_tasks x Sampled negative indices.
        """
        device = gt_labels_3d.device
        ### don't use the gravity_center, directly use the geometry_center
        # gt_bboxes_3d = torch.cat(
        #     (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        # ).to(device)
        gt_bboxes_3d = gt_bboxes_3d.to(device)

        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)

        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            num_bboxes = bbox_pred.shape[0]
            assign_results = self.assigner.assign(bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
            # label targets
            labels = gt_bboxes.new_full((num_bboxes,),
                                        num_classes,
                                        dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)
            # bbox_targets
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            bbox_weights = torch.zeros_like(bbox_pred)
            bbox_weights[pos_inds] = 1.0

            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks \
            = multi_apply(task_assign, pred_bboxes, pred_logits, task_boxes, task_classes, self.num_classes)

        return labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int]): num_tasks x Number of negative samples.
        """
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits
        )
        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list, task_bbox_targets_list, \
            task_bbox_weights_list = [], [], [], []

        for task_id in range(task_num):
            num_total_pos_task = sum((inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum((inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append([labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append(
                [labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append(
                [bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append(
                [bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])

        return (task_labels_list, task_labels_weight_list, task_bbox_targets_list,
                task_bbox_weights_list, num_total_pos_tasks, num_total_neg_tasks)

    def _loss_single_task(self,
                          pred_bboxes,
                          pred_logits,
                          labels_list,
                          labels_weights_list,
                          bbox_targets_list,
                          bbox_weights_list,
                          num_total_pos,
                          num_total_neg):
        """"Compute loss for single task.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            pred_bboxes (Tensor): (batch_size, num_query, 10)
            pred_logits (Tensor): (batch_size, num_query, task_classes)
            labels_list (list[Tensor]): batch_size x (num_query, )
            labels_weights_list (list[Tensor]): batch_size x (num_query, )
            bbox_targets_list(list[Tensor]): batch_size x (num_query, 9)
            bbox_weights_list(list[Tensor]): batch_size x (num_query, 10)
            num_total_pos: int
            num_total_neg: int
        Returns:
            loss_cls
            loss_bbox
        """
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)

        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)

        ### for ce loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor
        )

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.code_weights)[None, :]

        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss_single(self,
                    pred_bboxes,
                    pred_logits,
                    gt_bboxes_3d,
                    gt_labels_3d):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            pred_bboxes (list[Tensor]): num_tasks x [bs, num_query, 10].
            pred_logits (list(Tensor]): num_tasks x [bs, num_query, task_classes]
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        batch_size = pred_bboxes[0].shape[0]
        pred_bboxes_list, pred_logits_list = [], []
        for idx in range(batch_size):
            pred_bboxes_list.append([task_pred_bbox[idx] for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx] for task_pred_logits in pred_logits])
        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, pred_bboxes_list, pred_logits_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._loss_single_task,
            pred_bboxes,
            pred_logits,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg
        )

        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    @force_fp32(apply_to=('preds_dicts'))
    def forward(self, preds_dicts, label_dicts, return_dict_flag=False):
        """"Loss function.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                vel: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if 'fuse_ret_dicts' in preds_dicts.keys():
            ### single ###
            s_preds_dicts = preds_dicts['single_ego_ret_dicts']
            s_gt_bboxes_3d = label_dicts['single_ego_gt_bboxes_3d']
            s_gt_labels_3d = label_dicts['single_ego_gt_labels_3d']

            s_num_decoder = s_preds_dicts[0]['center'].shape[0]
            s_all_pred_bboxes, s_all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)

            for s_task_id, s_preds_dict in enumerate(s_preds_dicts, 0):
                for s_dec_id in range(s_num_decoder):
                    s_pred_bbox = torch.cat(
                        (s_preds_dict['center'][s_dec_id], s_preds_dict['height'][s_dec_id],
                         s_preds_dict['dim'][s_dec_id], s_preds_dict['rot'][s_dec_id]),  ### remove the velocity prediction
                        dim=-1
                    )
                    s_all_pred_bboxes[s_dec_id].append(s_pred_bbox)
                    s_all_pred_logits[s_dec_id].append(s_preds_dict['cls_logits'][s_dec_id])
            s_all_pred_bboxes = [s_all_pred_bboxes[s_idx] for s_idx in range(s_num_decoder)]
            s_all_pred_logits = [s_all_pred_logits[s_idx] for s_idx in range(s_num_decoder)]

            s_loss_cls, s_loss_bbox = multi_apply(
                self.loss_single, s_all_pred_bboxes, s_all_pred_logits,
                [s_gt_bboxes_3d for _ in range(s_num_decoder)],
                [s_gt_labels_3d for _ in range(s_num_decoder)],
            )
            s_total_loss = sum(s_loss_cls + s_loss_bbox)

            self.loss_dict['single_ego_total_loss'] = s_total_loss
            self.loss_dict['single_ego_loss_cls'] = s_loss_cls[-1]
            self.loss_dict['single_ego_loss_bbox'] = s_loss_bbox[-1]

            s_num_dec_layer = 0
            for s_loss_cls_i, s_loss_bbox_i in zip(s_loss_cls[:-1],
                                               s_loss_bbox[:-1]):
                self.loss_dict[f'd{s_num_dec_layer}.single_ego_loss_cls'] = s_loss_cls_i
                self.loss_dict[f'd{s_num_dec_layer}.single_ego_loss_bbox'] = s_loss_bbox_i
                s_num_dec_layer += 1


            ### fuse ###
            f_preds_dicts = preds_dicts['fuse_ret_dicts']
            f_gt_bboxes_3d = label_dicts['gt_bboxes_3d']
            f_gt_labels_3d = label_dicts['gt_labels_3d']

            f_num_decoder = f_preds_dicts[0]['center'].shape[0]
            f_all_pred_bboxes, f_all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)

            for f_task_id, f_preds_dict in enumerate(f_preds_dicts, 0):
                for f_dec_id in range(f_num_decoder):
                    f_pred_bbox = torch.cat(
                        (f_preds_dict['center'][f_dec_id], f_preds_dict['height'][f_dec_id],
                         f_preds_dict['dim'][f_dec_id], f_preds_dict['rot'][f_dec_id]),  ### remove the velocity prediction
                        dim=-1
                    )
                    f_all_pred_bboxes[f_dec_id].append(f_pred_bbox)
                    f_all_pred_logits[f_dec_id].append(f_preds_dict['cls_logits'][f_dec_id])
            f_all_pred_bboxes = [f_all_pred_bboxes[f_idx] for f_idx in range(f_num_decoder)]
            f_all_pred_logits = [f_all_pred_logits[f_idx] for f_idx in range(f_num_decoder)]

            f_loss_cls, f_loss_bbox = multi_apply(
                self.loss_single, f_all_pred_bboxes, f_all_pred_logits,
                [f_gt_bboxes_3d for _ in range(f_num_decoder)],
                [f_gt_labels_3d for _ in range(f_num_decoder)],
            )
            f_total_loss = sum(f_loss_cls + f_loss_bbox)

            self.loss_dict['fuse_total_loss'] = f_total_loss
            self.loss_dict['fuse_loss_cls'] = f_loss_cls[-1]
            self.loss_dict['fuse_loss_bbox'] = f_loss_bbox[-1]

            f_num_dec_layer = 0
            for f_loss_cls_i, f_loss_bbox_i in zip(f_loss_cls[:-1],
                                               f_loss_bbox[:-1]):
                self.loss_dict[f'd{f_num_dec_layer}.fuse_loss_cls'] = f_loss_cls_i
                self.loss_dict[f'd{f_num_dec_layer}.fuse_loss_bbox'] = f_loss_bbox_i
                f_num_dec_layer += 1

            ### total ###
            total_loss = sum(s_loss_cls + s_loss_bbox + f_loss_cls + f_loss_bbox)

            self.loss_dict['total_loss'] = total_loss

        else:
            preds_dicts = preds_dicts['single_ego_ret_dicts']
            gt_bboxes_3d = label_dicts['gt_bboxes_3d']
            gt_labels_3d = label_dicts['gt_labels_3d']

            num_decoder = preds_dicts[0]['center'].shape[0]
            all_pred_bboxes, all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)

            for task_id, preds_dict in enumerate(preds_dicts, 0):
                for dec_id in range(num_decoder):
                    pred_bbox = torch.cat(
                        (preds_dict['center'][dec_id], preds_dict['height'][dec_id],
                         preds_dict['dim'][dec_id], preds_dict['rot'][dec_id]),  ### remove the velocity prediction
                        dim=-1
                    )
                    all_pred_bboxes[dec_id].append(pred_bbox)
                    all_pred_logits[dec_id].append(preds_dict['cls_logits'][dec_id])
            all_pred_bboxes = [all_pred_bboxes[idx] for idx in range(num_decoder)]
            all_pred_logits = [all_pred_logits[idx] for idx in range(num_decoder)]

            loss_cls, loss_bbox = multi_apply(
                self.loss_single, all_pred_bboxes, all_pred_logits,
                [gt_bboxes_3d for _ in range(num_decoder)],
                [gt_labels_3d for _ in range(num_decoder)],
            )

            total_loss = sum(loss_cls + loss_bbox)

            self.loss_dict['total_loss'] = total_loss
            self.loss_dict['loss_cls'] = loss_cls[-1]
            self.loss_dict['loss_bbox'] = loss_bbox[-1]

            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1],
                                               loss_bbox[:-1]):
                self.loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
                self.loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
                num_dec_layer += 1

        if return_dict_flag:
            return total_loss, self.loss_dict
        else:
            return total_loss


    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        cls_loss = self.loss_dict['loss_cls']
        bbox_loss = self.loss_dict['loss_bbox']
        d0_cls_loss = self.loss_dict['d0.loss_cls']
        d0_bbox_loss= self.loss_dict['d0.loss_bbox']
        d1_cls_loss = self.loss_dict['d1.loss_cls']
        d1_bbox_loss= self.loss_dict['d1.loss_bbox']
        d2_cls_loss = self.loss_dict['d2.loss_cls']
        d2_bbox_loss= self.loss_dict['d2.loss_bbox']
        d3_cls_loss = self.loss_dict['d3.loss_cls']
        d3_bbox_loss= self.loss_dict['d3.loss_bbox']
        d4_cls_loss = self.loss_dict['d4.loss_cls']
        d4_bbox_loss= self.loss_dict['d4.loss_bbox']


        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), cls_loss.item(), bbox_loss.item()))
        else:
            print("[epoch %d][%d/%d], || Loss: %.4f || Cls Loss: %.4f"
                " || Bbox Loss: %.4f" 
                " || d0_cls_loss: %.4f || d0_bbox_loss: %.4f"
                " || d1_cls_loss: %.4f || d1_bbox_loss: %.4f"
                " || d2_cls_loss: %.4f || d2_bbox_loss: %.4f"
                " || d3_cls_loss: %.4f || d3_bbox_loss: %.4f"
                " || d4_cls_loss: %.4f || d4_bbox_loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), cls_loss.item(), bbox_loss.item(),
                d0_cls_loss.item(), d0_bbox_loss.item(),
                d1_cls_loss.item(), d1_bbox_loss.item(),
                d2_cls_loss.item(), d2_bbox_loss.item(),
                d3_cls_loss.item(), d3_bbox_loss.item(),
                d4_cls_loss.item(), d4_bbox_loss.item(),
                    ))

        writer.add_scalar('total_loss', total_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('cls_loss', cls_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('bbox_loss', bbox_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('d0_cls_loss', d0_cls_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('d0_bbox_loss', d0_bbox_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('d1_cls_loss', d1_cls_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('d1_bbox_loss', d1_bbox_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('d2_cls_loss', d2_cls_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('d2_bbox_loss', d2_bbox_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('d3_cls_loss', d3_cls_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('d3_bbox_loss', d3_bbox_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('d4_cls_loss', d4_cls_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('d4_bbox_loss', d4_bbox_loss.item(),
                          epoch*batch_len + batch_id)


if __name__ == '__main__':
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils
    hypes = load_yaml('/media/re/2384a6b4-4dae-400d-ad72-9b7044491b55/V2V_PR1_Camera/QB-V2V/opencood/hypes_yaml/CMT/camera/cmt_loss.yaml')
    criterion = train_utils.create_loss(hypes)
    print(criterion)
    pass