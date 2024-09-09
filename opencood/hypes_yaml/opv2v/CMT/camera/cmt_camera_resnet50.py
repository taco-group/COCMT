plugin=True
plugin_dir='projects/mmdet3d_plugin/'

with_cp = True

point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 1]

voxel_size = [0.40, 0.40, 4]

model = dict(
    type='CmtDetector_PointPillar',
    use_grid_mask=False,
    img_backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=with_cp,
        style='pytorch'),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    pts_bbox_head=dict(
        type='CmtImageHead',
        num_query=900,
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        transformer=dict(
            type='CmtImageTransformer',
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
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)),
)