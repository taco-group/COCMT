import torch.nn as nn
import torch.onnx.symbolic_opset8

from opencood.models.cmt_camera import CMTCamera
from opencood.models.cmt_lidar import CMTLiDAR
from opencood.models.fuse_modules.fuse_utils import regroup_query_or_reference_points

### COMLN
from opencood.models.mmdet3d_plugin.models.utils.mln_utils import MLN, transform_object_centers, nerf_positional_encoding, pos2posemb3d
### sigmoid
from mmdet.models.utils.transformer import inverse_sigmoid

### FuseNet
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence


### co_task head
from mmdet3d.models import builder
from opencood.models.mmdet3d_plugin import *
import copy

### bboxcoder
from mmdet3d.core import bbox3d2result

class CMTCameraLidarAttFuse(nn.Module):

    def __init__(self, config):
        super(CMTCameraLidarAttFuse, self).__init__()
        self.fused_ego_detection_range = nn.Parameter(torch.tensor(config['fused_ego_detection_range']), requires_grad=False)

        if config['camera']['use_camera_flag']:
            self.camera_encoder = CMTCamera(config['camera'])
            self.camera_encoder.set_return_features()
            self.single_camera_cav_detection_range = nn.Parameter(torch.tensor(config['single_camera_cav_detection_range']), requires_grad=False)
            self.max_cav = config['camera']['max_cav']
        if config['lidar']['use_lidar_flag']:
            self.lidar_encoder = CMTLiDAR(config['lidar'])
            self.lidar_encoder.set_return_features()
            self.single_lidar_cav_detection_range = nn.Parameter(torch.tensor(config['single_lidar_cav_detection_range']), requires_grad=False)
            self.max_cav = config['lidar']['max_cav']

        ### COMLN
        self.use_MLN = config['use_MLN']
        if self.use_MLN:
            self.cav_pose_encode_query = MLN(144)

        ### FuseNet
        self.fusion_net = build_transformer_layer_sequence(config['Fuser'])
        self.fusion_net.init_weights()
        self.num_heads = config['Fuser'][ 'transformerlayers']['attn_cfgs']['num_heads']
        self.embed_dims = config['Fuser'][ 'transformerlayers']['attn_cfgs']['embed_dims']
        self.distance_mask_flag = config['distance_mask']['flag']
        self.mask_distance = config['distance_mask']['distance']
        self.object_mask_flag = config['object_mask']['flag']
        self.mask_score = config['object_mask']['score_threshold']

        ### cooperate head
        co_separate_head_cfg = config['CO_SeperateHead']
        common_heads = co_separate_head_cfg['common_heads']
        num_cls = co_separate_head_cfg['num_cls']
        heads = copy.deepcopy(common_heads)
        heads.update(dict(cls_logits=[num_cls, 2]))
        co_separate_head_cfg.update(heads=heads)

        self.co_task_heads = nn.ModuleList()
        self.co_task_heads.append(builder.build_head(co_separate_head_cfg))
        self.co_task_heads[0].init_weights()

        self._fix_camera_backbone = False
        self._fix_lidar_backbone = False

    def reformat_input(self, agent_modality_list, record_len, max_cav):
        # 获取总记录数
        total_records = record_len.sum().item()
        # 初始化 mode 张量
        mode = torch.zeros((len(record_len), max_cav), device=record_len.device, dtype=torch.long)
        # 当前记录的索引
        current_index = 0
        # 遍历每个记录的长度
        for i, length in enumerate(record_len):
            length = length.item()
            # 遍历当前记录的每个元素
            for j in range(length):
                # 根据 agent_modality_list 的值设置 mode 张量
                if agent_modality_list[current_index] == 'm1':
                    mode[i, j] = 1
                elif agent_modality_list[current_index] == 'm2':
                    mode[i, j] = 0
                current_index += 1
        return mode

    def unpad_mode_encoding(self, mode, record_len):
        B = mode.shape[0]
        out = []
        for i in range(B):
            out.append(mode[i, :record_len[i]])
        return torch.cat(out, dim=0)

    def combine_features(self, camera_feature, lidar_feature, mode,
                         record_len):
        combined_features = []
        if len(mode.shape) == 2:
            mode = self.unpad_mode_encoding(mode, record_len)
        camera_count = 0
        lidar_count = 0
        for i in range(len(mode)):
            if mode[i] == 0:
                combined_features.append(camera_feature[camera_count, ...])
                camera_count += 1
            elif mode[i] == 1:
                combined_features.append(lidar_feature[lidar_count, ...])
                lidar_count += 1
            else:
                raise ValueError(f"Mode but be either 1 or 0 but received "
                                 f"{mode[i]}")
        # print(f"Lidar/camera = {lidar_count} / {camera_count}")
        combined_features = torch.stack(combined_features, dim=0)
        return combined_features

    def fix_camera_backbone(self):
        self._fix_camera_backbone = True

    def fix_lidar_backbone(self):
        self._fix_lidar_backbone = True

    def _freeze_weights(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, batch, test_flag=False):
        if self._fix_lidar_backbone:
            self._freeze_weights(self.lidar_encoder)
        if self._fix_camera_backbone:
            self._freeze_weights((self.camera_encoder))

        max_cav = self.max_cav
        record_len = batch['record_len']
        # (B, L)
        mode = self.reformat_input(batch['agent_modality_list'], record_len, max_cav)
        mode_unpack = self.unpad_mode_encoding(mode, record_len)

        camera_features = None
        camera_reference_points = None
        camera_object_centers = None
        camera_cav_ret_dicts = None
        camera_object_scores = None
        lidar_features = None
        lidar_reference_points = None
        lidar_object_centers = None
        lidar_cav_ret_dicts = None
        lidar_object_scores = None

        transformed_all_cav_ocs = None

        # If there is at least one camera
        if not torch.all(mode_unpack == 1):
            camera_cav_ret_dicts, camera_features, camera_reference_points, camera_object_centers, camera_object_scores = self.camera_encoder(batch)
            ### denormalize the camera_reference_points at single camera cav detection range
            camera_reference_points = camera_reference_points * (self.single_camera_cav_detection_range[3:6] - self.single_camera_cav_detection_range[0:3]) + self.single_camera_cav_detection_range[0:3]
            ### normalize the camera_reference_points at fused detection range
            camera_reference_points = (camera_reference_points - self.fused_ego_detection_range[0:3]) / (self.fused_ego_detection_range[3:6] - self.fused_ego_detection_range[0:3])

        # If there is at least one lidar
        if not torch.all(mode_unpack == 0):
            lidar_cav_ret_dicts, lidar_features, lidar_reference_points, lidar_object_centers, lidar_object_scores = self.lidar_encoder(batch)
            lidar_reference_points = lidar_reference_points * (self.single_lidar_cav_detection_range[3:6] - self.single_lidar_cav_detection_range[0:3]) + self.single_lidar_cav_detection_range[0:3]
            lidar_reference_points = (lidar_reference_points - self.fused_ego_detection_range[0:3]) / (self.fused_ego_detection_range[3:6] - self.fused_ego_detection_range[0:3])

        with torch.cuda.amp.autocast(enabled=False):
            ### start fuse ###

            # ### reference --- to get ego reference
            # # N, Q, 3
            # all_cav_reference_points = self.combine_features(camera_reference_points,
            #                                   lidar_reference_points, mode_unpack,
            #                                   record_len)
            # # B, L, Q, 3
            # all_cav_reference_points, _ = regroup_query_or_reference_points(all_cav_reference_points, record_len, max_cav)
            # # B, 1, Q, 3
            # ego_reference_points = all_cav_reference_points[:, 0:1, :, :]
            # cavs_reference_points = all_cav_reference_points[:, 1:, :, :]

            ### query feature --- to get ego cavs object query & other cavs object query
            all_cav_object_query = self.combine_features(camera_features,
                                      lidar_features, mode_unpack,
                                      record_len)

            # N, Q, C -> B, L, Q, C     B, L
            all_cav_object_query, cav_mask = regroup_query_or_reference_points(all_cav_object_query, record_len, max_cav)
            B, L, Q, C = all_cav_object_query.shape

            ego_object_query = all_cav_object_query[:, 0:1, :, :]
            cavs_object_query = all_cav_object_query[:, 1:, :, :]


            ''' MLN '''
            if self.use_MLN:
                ### co_reference --- to get other cavs predicted 3D object centers
                all_cav_object_centers = self.combine_features(camera_object_centers,
                                                  lidar_object_centers, mode_unpack,
                                                  record_len)
                # B, L, Q, 3
                all_cav_object_centers, _ = regroup_query_or_reference_points(all_cav_object_centers, record_len, max_cav)
                ego_object_centers = all_cav_object_centers[:, 0:1, :, :]
                ego_reference_points = (ego_object_centers - self.fused_ego_detection_range[0:3]) / (self.fused_ego_detection_range[3:6] - self.fused_ego_detection_range[0:3])
                # B, L-1, Q, 3
                cavs_object_centers = all_cav_object_centers[:, 1:, :, :]

                # transformation matrix -> to get other cavs transformation matrix from (other cavs) to (ego vehicle)
                transformation_matrix = batch['transformation_matrix_full'] # B, L, 4, 4
                cavs2ego = transformation_matrix[:, 1:, :, :] # B, L-1, 4, 4

                # TODO: this seems have some bug!!!
                ''' Equation 2'''
                ### transform (other cavs predicted 3D object centers) from (other cavs coordinate system) to (ego vehicle coordinate system)
                cavs_reference_points = transform_object_centers(cavs_object_centers, cavs2ego)
                transformed_all_cav_ocs = torch.cat((ego_object_centers, cavs_reference_points), dim=1)  # B, L, Q, 3
                ### normalize other cavs predicted 3D object centers at (ego vehicle coordinate system)
                cavs_reference_points = (cavs_reference_points - self.fused_ego_detection_range[0:3]) / (self.fused_ego_detection_range[3:6] - self.fused_ego_detection_range[0:3])

                ''' Equation 3-1 3-2 4'''
                cavs2ego = cavs2ego.unsqueeze(2).repeat(1, 1, Q, 1, 1) # B, L-1, 4, 4 -> B, L-1, Q, 4, 4
                cavs_motion = cavs2ego[..., :3, :].flatten(-2) # B, L-1, Q, 3, 4 -> B, L-1, Q, 12
                cavs_motion = nerf_positional_encoding(cavs_motion) # B, L-1, Q, 144
                cavs_object_query = self.cav_pose_encode_query(cavs_object_query, cavs_motion) # B, L-1, Q, C


            ### group (original ego vehicle information) and (other cavs information after MLN)
            x = torch.cat((ego_object_query, cavs_object_query), dim=1) # B, L, Q, C

            fused_reference_points = torch.cat((ego_reference_points, cavs_reference_points), dim=1) # B, L, Q, 3

            ### reference_points : B, L, Q, 3 -> B, L*Q, 3
            fused_reference_points = fused_reference_points.reshape(B, L*Q, 3)

            ### x : B, L, Q, C -> B, L*Q, C
            x = x.reshape(B, L*Q, C)


            ### build cav_mask to block out padded invalid cavs
            cav_mask = cav_mask.unsqueeze(-1)  # 添加一个新的维度用于复制
            cav_mask = cav_mask.repeat(1, 1, Q)  # 在最后一个维度上复制 Q 次
            cav_mask = cav_mask.reshape(B, -1)  # 重塑 mask 以匹配 (B, L*Q)
            extended_cav_mask = cav_mask.unsqueeze(1).expand(B, L*Q, L*Q)  # B, L*Q, L*Q
            cav_mask = cav_mask.unsqueeze(2) # B, L*Q, 1
            ### CAV Mask
            cav_mask = extended_cav_mask * cav_mask  # B, L*Q, L*Q

            if self.distance_mask_flag and self.use_MLN:

                ### build distance_mask to allow only close-distance cavs (object query) to interact with each other
                transformed_all_cav_ocs = transformed_all_cav_ocs.reshape(B, L*Q, 3) # (B, L*Q, 3)
                transformed_all_cav_ocs_expanded = transformed_all_cav_ocs.unsqueeze(2) # (B, L*Q, 1, 3)
                transformed_all_cav_ocs = transformed_all_cav_ocs.unsqueeze(1) # (B, L*Q, 3, 1)
                distances = torch.sqrt(((transformed_all_cav_ocs_expanded - transformed_all_cav_ocs) ** 2).sum(-1)) # (B, L*Q, L*Q)
                dist_mask = (distances < self.mask_distance)

                ### Distance Mask
                tmp_mask = dist_mask & cav_mask

            if self.object_mask_flag:
                all_cav_object_scores = self.combine_features(camera_object_scores,
                                                  lidar_object_scores, mode_unpack,
                                                  record_len)
                # B, L, Q, 1
                all_cav_object_scores, _ = regroup_query_or_reference_points(all_cav_object_scores, record_len, max_cav)


                object_mask = all_cav_object_scores.reshape(B, L*Q) > self.mask_score
                extended_object_mask = object_mask.unsqueeze(1).expand(B, L*Q, L*Q)  # B, L*Q, L*Q
                object_mask = object_mask.unsqueeze(2) # B, L*Q, 1
                object_mask = extended_object_mask * object_mask  # B, L*Q, L*Q

                ### Obejct Mask
                tmp_mask = object_mask & tmp_mask

            ### build eye_mask to meet the operational requirements of self-attention
            eye_mask = torch.eye(L * Q).long().to(cav_mask.device)
            eye_mask = eye_mask.unsqueeze(0).expand(B, L * Q, L * Q)

            ###
            mask = tmp_mask | eye_mask

            selfattn_mask = 1 - mask
            selfattn_mask = selfattn_mask.bool() # 使用astype函数将1变为True，将0变为False
            selfattn_mask = selfattn_mask.repeat_interleave(self.num_heads, dim=0)

            ### prepare for TransformerEncoder
            x = x.permute(1, 0, 2)

        ### fuse
        x = self.fusion_net(
            query=x,
            key=x,
            value=x,
            query_pos=None,
            key_pos=None,
            query_key_padding_mask=None,
            attn_masks=[selfattn_mask],
        )
        x = torch.nan_to_num(x)
        x = x.transpose(1, 2)

        ### fuse finish ###

        with torch.cuda.amp.autocast(enabled=False):

            ### sigmoid
            reference = inverse_sigmoid(fused_reference_points.clone())

            ret_dicts = []

            for task_id, task in enumerate(self.co_task_heads, 0):
                outs = task(x)
                center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
                height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
                _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
                _center[..., 0:1] = center[..., 0:1] * (self.fused_ego_detection_range[3] - self.fused_ego_detection_range[0]) + self.fused_ego_detection_range[0]
                _center[..., 1:2] = center[..., 1:2] * (self.fused_ego_detection_range[4] - self.fused_ego_detection_range[1]) + self.fused_ego_detection_range[1]
                _height[..., 0:1] = height[..., 0:1] * (self.fused_ego_detection_range[5] - self.fused_ego_detection_range[2]) + self.fused_ego_detection_range[2]
                outs['center'] = _center
                outs['height'] = _height

                ret_dicts.append(outs)

            ### multi range label predicts selection
            all_ret_dicts = {}
            all_ret_dicts['fuse_ret_dicts'] = ret_dicts
            # camera-only intermediate
            if torch.all(mode_unpack == 0):
                ego_record_len = record_len
                ego_cav_indexes = torch.zeros_like(ego_record_len)
                ego_cav_indexes[1:] = ego_record_len[:-1].cumsum(dim=0)
                for key in camera_cav_ret_dicts[0].keys():
                    camera_cav_ret_dicts[0][key] = camera_cav_ret_dicts[0][key][:, ego_cav_indexes]
                all_ret_dicts['single_ego_ret_dicts'] = camera_cav_ret_dicts
            # lidar-only intermediate
            elif torch.all(mode_unpack == 1):
                ego_record_len = record_len
                ego_cav_indexes = torch.zeros_like(ego_record_len)
                ego_cav_indexes[1:] = ego_record_len[:-1].cumsum(dim=0)
                for key in lidar_cav_ret_dicts[0].keys():
                    lidar_cav_ret_dicts[0][key] = lidar_cav_ret_dicts[0][key][:, ego_cav_indexes]
                all_ret_dicts['single_ego_ret_dicts'] = lidar_cav_ret_dicts
            # ego-camera hetero intermediate
            elif not torch.all(mode_unpack == 0) and torch.all(mode[:, 0] == 0):
                lidar_cav_count = mode.sum(dim=1).long()
                ego_record_len = record_len - lidar_cav_count
                ego_cav_indexes = torch.zeros_like(ego_record_len)
                ego_cav_indexes[1:] = ego_record_len[:-1].cumsum(dim=0)
                for key in camera_cav_ret_dicts[0].keys():
                    camera_cav_ret_dicts[0][key] = camera_cav_ret_dicts[0][key][:, ego_cav_indexes]
                all_ret_dicts['single_ego_ret_dicts'] = camera_cav_ret_dicts
            # ego-lidar hetero intermediate
            elif not torch.all(mode_unpack == 0) and torch.all(mode[:, 0] == 1):
                lidar_cav_count = mode.sum(dim=1).long()
                ego_record_len = lidar_cav_count
                ego_cav_indexes = torch.zeros_like(ego_record_len)
                ego_cav_indexes[1:] = ego_record_len[:-1].cumsum(dim=0)
                for key in lidar_cav_ret_dicts[0].keys():
                    lidar_cav_ret_dicts[0][key] = lidar_cav_ret_dicts[0][key][:, ego_cav_indexes]
                all_ret_dicts['single_ego_ret_dicts'] = lidar_cav_ret_dicts
            # ego-mixed hetero intermediate
            elif not torch.all(mode_unpack == 0) and not torch.all(mode[:, 0] == 1):
                ### 得到 mixed_ego results
                mixed_cav_ret_dicts = {}
                # 计算 lidar 和 camera 的 cav 数量
                lidar_cav_count = mode.sum(dim=1).long()
                camera_cav_count = record_len - lidar_cav_count

                # 计算 lidar 和 camera 的 cav 索引
                lidar_cav_indexes = torch.zeros_like(lidar_cav_count)
                camera_cav_indexes = torch.zeros_like(camera_cav_count)
                lidar_cav_indexes[1:] = lidar_cav_count[:-1].cumsum(dim=0)
                camera_cav_indexes[1:] = camera_cav_count[:-1].cumsum(dim=0)

                # get the mixed ego cav_ret_dicts
                for i in range(B):
                    if mode[i, 0] == 0:
                        # 处理 camera 数据
                        for key in camera_cav_ret_dicts[0].keys():
                            camera_cav_ret_dicts[0][key] = camera_cav_ret_dicts[0][key][:, camera_cav_indexes[i]]
                        mixed_cav_ret_dicts[i] = camera_cav_ret_dicts
                    elif mode[i, 0] == 1:
                        # 处理 lidar 数据
                        for key in lidar_cav_ret_dicts[0].keys():
                            lidar_cav_ret_dicts[0][key] = lidar_cav_ret_dicts[0][key][:, lidar_cav_indexes[i]]
                        mixed_cav_ret_dicts[i] = lidar_cav_ret_dicts

                ### 将 mixed_ego_results 合并
                # 初始化一个空字典来存储合并后的结果
                merged_dict = {key: [] for key in mixed_cav_ret_dicts[0][0].keys()}

                # 遍历 mixed_cav_ret_dicts 的所有键和值
                for outer_key, value_dict in mixed_cav_ret_dicts.items():
                    for key, value in value_dict[0].items():
                        merged_dict[key].append(value)

                # 将列表中的张量在第二个维度拼接起来，并调整维度
                for key, value_list in merged_dict.items():
                    # 确保 value_list 是一个包含张量的列表
                    if isinstance(value_list, list) and all(isinstance(v, torch.Tensor) for v in value_list):
                        # 在第二个维度添加一个新的维度
                        expanded_tensors = [v.unsqueeze(1) for v in value_list]
                        # 在第二个维度拼接张量
                        merged_dict[key] = torch.cat(expanded_tensors, dim=1)
                    else:
                        raise ValueError(f"Expected value_list to be a list of tensors, but got {type(value_list)}")

                all_ret_dicts['single_ego_ret_dicts'] = [merged_dict]

            if test_flag is not True:
                # train
                return all_ret_dicts
            else:
                # test
                bbox_list = self.co_task_heads[0].get_bboxes(
                    ret_dicts, img_metas=None, rescale=False)
                bbox_results = [
                    bbox3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels, _ in bbox_list
                ]
                bbox_results[0]['query_index'] = bbox_list[0][3].cpu()
                return bbox_results[0]