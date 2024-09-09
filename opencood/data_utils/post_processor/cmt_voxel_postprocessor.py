### from voxel_postprocessor.py
"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
from torch.nn.functional import sigmoid
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils

class CMT_VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(CMT_VoxelPostprocessor, self).__init__(anchor_params, train)

    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        ### 增加 lwh
        assert self.params['order'] == 'hwl' or 'lwh', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        multi_range_label_flag = kwargs.get('multi_range_label_flag', False)
        if multi_range_label_flag:
            # (max_num, 7)
            gt_box_center = kwargs['gt_box_center']

            # (max_num)
            masks = kwargs['mask']

            # (n, 7)
            gt_box_center_valid = gt_box_center[masks == 1]

            ### 强制类型转换
            gt_box_center_valid = gt_box_center_valid.astype('float32')

            ### build gt labels
            gt_label = masks[masks == 1]
            ### !!! align for nuscenes label -> nuscenes label start by 0 -> opv2v label start by 1
            gt_label = gt_label - 1
            ### 做强制类型转换 for issue11
            gt_label = gt_label.astype(int)

            label_dict = {'single_ego_gt_bboxes_3d': gt_box_center_valid,
                          'single_ego_gt_labels_3d': gt_label}
        else:
            # (max_num, 7)
            gt_box_center = kwargs['gt_box_center']

            # (max_num)
            masks = kwargs['mask']

            # (n, 7)
            gt_box_center_valid = gt_box_center[masks == 1]

            ### 强制类型转换
            gt_box_center_valid = gt_box_center_valid.astype('float32')

            ### build gt labels
            gt_label = masks[masks == 1]
            ### !!! align for nuscenes label -> nuscenes label start by 0 -> opv2v label start by 1
            gt_label = gt_label - 1
            ### 做强制类型转换 for issue11
            gt_label = gt_label.astype(int)

            label_dict = {'gt_bboxes_3d': gt_box_center_valid,
                          'gt_labels_3d': gt_label}

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        if 'single_ego_gt_bboxes_3d' in label_batch_list[0].keys():
            gt_bboxes_3d = []
            gt_labels_3d = []
            single_ego_gt_bboxes_3d = []
            single_ego_gt_labels_3d = []

            for i in range(len(label_batch_list)):
                gt_bboxes_3d.append(torch.tensor(label_batch_list[i]['gt_bboxes_3d']))
                gt_labels_3d.append(torch.tensor(label_batch_list[i]['gt_labels_3d']))
                single_ego_gt_bboxes_3d.append(torch.tensor(label_batch_list[i]['single_ego_gt_bboxes_3d']))
                single_ego_gt_labels_3d.append(torch.tensor(label_batch_list[i]['single_ego_gt_labels_3d']))

            return {'gt_bboxes_3d': gt_bboxes_3d,
                    'gt_labels_3d': gt_labels_3d,
                    'single_ego_gt_bboxes_3d': single_ego_gt_bboxes_3d,
                    'single_ego_gt_labels_3d': single_ego_gt_labels_3d}

        else:
            gt_bboxes_3d = []
            gt_labels_3d = []

            for i in range(len(label_batch_list)):
                gt_bboxes_3d.append(torch.tensor(label_batch_list[i]['gt_bboxes_3d']))
                gt_labels_3d.append(torch.tensor(label_batch_list[i]['gt_labels_3d']))

            return {'gt_bboxes_3d': gt_bboxes_3d,
                    'gt_labels_3d': gt_labels_3d}

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        for cav_id in output_dict.keys():
            assert cav_id in data_dict
            cav_content = data_dict[cav_id]
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix'].to('cpu') # no clean

            # (H, W, anchor_num, 7)
            # anchor_box = cav_content['anchor_box']

            # classification probability ### unsqueeze(0) 仿照 batchsize
            prob = output_dict[cav_id]['scores_3d'].unsqueeze(0)
            # prob = output_dict[cav_id]['psm']
            # prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            # prob = prob.reshape(1, -1)

            # regression map
            # reg = output_dict[cav_id]['rm']

            # convert regression map back to bounding box
            # (N, W*L*anchor_num, 7)
            # batch_box3d = self.delta_to_boxes3d(reg, anchor_box) ### unsqueeze(0) 仿照 batchsize
            batch_box3d = output_dict[cav_id]['boxes_3d'].unsqueeze(0)

            mask = \
                torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])

                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # filter out the prediction out of the range. with z-dim
        pred_box3d_np = pred_box3d_tensor.cpu().numpy()
        pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(pred_box3d_np,
                                                    self.params['gt_range'],
                                                    order=None,
                                                    return_mask=True)
        pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device)
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)
