plugin=True
plugin_dir='projects/mmdet3d_plugin/'

with_cp = True

point_cloud_range = [-102.4, -102.4, -3, 102.4, 102.4, 1]

grid_size = [512, 512, 1]

voxel_size = [ 0.4, 0.4, 4 ]

model = dict(
    type='CmtDetector_PointPillar',
    lidar_backbone=dict(
        pillar_vfe=dict(
            use_norm=True,
            with_distance=False,
            use_absolute_xyz=True,
            num_filters=[64]
        ),
        point_pillar_scatter=dict(
            num_features=64,
            grid_size=grid_size
        ),
        base_bev_backbone=dict(
            layer_nums=[3, 5, 8],
            layer_strides=[2, 2, 2],
            num_filters=[64, 128, 256],
            upsample_strides=[1, 2, 4],
            num_upsample_filter=[128, 128, 128]
        ),
        # resnet_bev_backbone=dict(
        #     layer_nums=[3, 5, 8],
        #     layer_strides=[2, 2, 2],
        #     num_filters=[64, 128, 256],
        #     upsample_strides=[1, 2, 4],
        #     num_upsample_filter=[128, 128, 128]
        # ),
        shrink_header=dict(
            kernal_size=[3],
            stride=[2],
            padding=[1],
            dim=[256],
            input_dim=384  # 128 * 3
        ),
    ),
    pts_bbox_head=dict(
        type='CmtLidarHead',
        in_channels=256,
        hidden_dim=256,
        downsample_scale=4,
        transformer=dict(
            type='CmtLidarTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=with_cp,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=1024, #unused
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            ))),
    train_cfg=dict(
        pts=dict(
            assigner=dict(type='HungarianAssigner3D'),### fake code, just for build the pts_head
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)),
)
