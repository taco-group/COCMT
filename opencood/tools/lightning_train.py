import argparse

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.utils import cmt_utils
from opencood.data_utils.datasets import build_dataset

import lightning as L
from pytorch_lightning import seed_everything

from opencood.utils.custom_lr import WarmupCosineAnnealingLR
from opencood.utils.cmt_utils import load_cmt_camera_nofusion_checkpoint, load_cmt_lidar_nofusion_checkpoint, load_ckpt

import os
import logging

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for training')
    parser.add_argument('--fix_camera_backbone', action='store_true',
                        help='fix the parameters of camera backbone')
    parser.add_argument('--fix_lidar_backbone', action='store_true',
                        help='fix the parameters of camera backbone')
    parser.add_argument('--load_cmt_camera_nofusion_weights', action='store_true',
                        help='load_cmt_camera_nofusion_weights for intermediatefusion')
    parser.add_argument('--load_cmt_lidar_nofusion_weights', action='store_true',
                        help='load_cmt_lidar_nofusion_weights for intermediatefusion')
    parser.add_argument('--cmt_camera_nofusion_weights_path', type=str,
                        help='cmt_camera_nofusion_weights_path')
    parser.add_argument('--cmt_lidar_nofusion_weights_path', type=str,
                        help='cmt_lidar_nofusion_weights_path')
    parser.add_argument('--ckpt_path', type=str,
                        help='lightning ckpt path')
    opt = parser.parse_args()
    return opt

class MyModel(L.LightningModule):
    def __init__(self, hypes):
        super().__init__()
        self.hypes = hypes
        self.save_hyperparameters(hypes)
        self.model = train_utils.create_model(hypes)
        # define the lowarmup + cosine annealss
        self.loss = train_utils.create_loss(hypes)

    def setup(self, stage=None):
        if 'reproduce_the_experiment' in self.hypes:
            seeds = self.hypes['reproduce_the_experiment']['seed_list']
            saved_path = self.hypes['saved_path']
            # 获取当前GPU的rank
            local_rank = self.trainer.local_rank if self.trainer else 0
            # 根据local_rank选择种子
            if local_rank < len(seeds):
                seed = seeds[local_rank]
            else:
                raise ValueError(f"Local rank {local_rank} out of range. Only {len(seeds)} seeds provided.")
            # 设置随机种子并获取它
            actual_seed = seed_everything(seed)
            # 打印当前的local_rank和设置的种子，以便验证
            print(f"Local rank: {local_rank}, Seed set to: {actual_seed}")
            # 记录单独的 log 文件
            logging.basicConfig(filename=os.path.dirname(saved_path) + '/set_seed_by_ranks.log', level=logging.INFO)
            logging.info(saved_path + ' Set seed to %d' % actual_seed)

    def training_step(self, batch, batch_idx):
        ouput_dict = self.model(batch['ego'])
        loss, loss_dict= self.loss(ouput_dict, batch['ego']['label_dict'], return_dict_flag=True)
        self.lightning_log(loss_dict)
        return loss

    def configure_optimizers(self):
        base_lr = self.hypes['optimizer']['lr']
        if hasattr(self.model, 'None'):
            ### 将模型参数组分开, 然后添加倍率
            optimizer_grouped_parameters = [{'params': self.model.camera_encoder.parameters(), 'multiplier': 0.01, 'name': 'camera_encoder'},
                                            {'params': self.model.fusion_net.parameters(), 'multiplier': 1, 'name': 'fusion_net'},
                                            {'params': self.model.task_heads.parameters(), 'multiplier': 1, 'name': 'task_heads'}]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          eps=self.hypes['optimizer']['args']['eps'],
                                          weight_decay=self.hypes['optimizer']['args']['weight_decay'])
            lr_scheduler = WarmupCosineAnnealingLR(optimizer,
                                                   warmup_lr=self.hypes['lr_scheduler']['warmup_lr'],
                                                   max_lr=self.hypes['optimizer']['lr'],
                                                   min_lr=self.hypes['lr_scheduler']['lr_min'],
                                                   warmup_epoch=self.hypes['lr_scheduler']['warmup_epoches'],
                                                   max_epoch=self.hypes['lr_scheduler']['epoches'])
        else:
            optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=base_lr,
                                          eps=self.hypes['optimizer']['args']['eps'],
                                          weight_decay=self.hypes['optimizer']['args']['weight_decay'])
            lr_scheduler = WarmupCosineAnnealingLR(optimizer,
                                                   warmup_lr=self.hypes['lr_scheduler']['warmup_lr'],
                                                   max_lr=self.hypes['optimizer']['lr'],
                                                   min_lr=self.hypes['lr_scheduler']['lr_min'],
                                                   warmup_epoch=self.hypes['lr_scheduler']['warmup_epoches'],
                                                   max_epoch=self.hypes['lr_scheduler']['epoches'])
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        ouput_dict = self.model(batch['ego'])
        loss, loss_dict = self.loss(ouput_dict, batch['ego']['label_dict'], return_dict_flag=True)
        if 'fuse_total_loss' in loss_dict.keys():
            self.log('val_loss', loss_dict['fuse_total_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hypes['train_params']['batch_size'])
            total_loss = loss_dict['total_loss']
            s_total_loss = loss_dict['single_ego_total_loss']
            f_total_loss = loss_dict['fuse_total_loss']
            print("VAL Total Loss: %.4f || VAL Single EGO Total Loss: %.4f || VAL Fuse Total Loss: %.4f " % (total_loss.item(), s_total_loss.item(), f_total_loss.item()))
        else:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hypes['train_params']['batch_size'])
            print("VAL Total Loss: %.4f " % ( loss.item() ))
        return loss

    def lightning_log(self, loss_dict):
        if 'fuse_total_loss' in loss_dict.keys():
            total_loss = loss_dict['total_loss']
            s_total_loss = loss_dict['single_ego_total_loss']
            f_total_loss = loss_dict['fuse_total_loss']
            s_cls_loss = loss_dict['single_ego_loss_cls']
            s_bbox_loss = loss_dict['single_ego_loss_bbox']
            f_cls_loss = loss_dict['fuse_loss_cls']
            f_bbox_loss = loss_dict['fuse_loss_bbox']

            print("Total Loss: %.4f || Single EGO Total Loss: %.4f || Fuse Total Loss: %.4f || Single EGO Cls Loss: %.4f || Single EGO BBox Loss: %.4f || Fuse Cls Loss: %.4f || Fuse BBox Loss: %.4f" % (total_loss.item(), s_total_loss.item(), f_total_loss.item(), s_cls_loss.item(), s_bbox_loss.item(), f_cls_loss.item(), f_bbox_loss.item()))

            self.log('total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.hypes['train_params']['batch_size'])
            self.log('single_ego_total_loss', s_total_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.hypes['train_params']['batch_size'])
            self.log('fuse_total_loss', f_total_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.hypes['train_params']['batch_size'])
            self.log('single_ego_cls_loss', s_cls_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.hypes['train_params']['batch_size'])
            self.log('single_ego_bbox_loss', s_bbox_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.hypes['train_params']['batch_size'])
            self.log('fuse_cls_loss', f_cls_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.hypes['train_params']['batch_size'])
            self.log('fuse_bbox_loss', f_bbox_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.hypes['train_params']['batch_size'])
        else:
            total_loss = loss_dict['total_loss']
            cls_loss = loss_dict['loss_cls']
            bbox_loss = loss_dict['loss_bbox']

            print("Loss: %.4f || Cls Loss: %.4f || BBox Loss: %.4f" % (total_loss.item(), cls_loss.item(), bbox_loss.item()))

            self.log('total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True,
                     batch_size=self.hypes['train_params']['batch_size'])
            self.log('cls_loss', cls_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True,
                     batch_size=self.hypes['train_params']['batch_size'])
            self.log('bbox_loss', bbox_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True,
                     batch_size=self.hypes['train_params']['batch_size'])

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    saved_path = train_utils.setup_train_with_mmcv(hypes)
    hypes['saved_path'] = saved_path

    ### call backs
    from lightning.pytorch.callbacks import ModelCheckpoint
    # 创建 ModelCheckpoint 回调，配置为保存所有周期的检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=saved_path,
        filename='{epoch:01d}',
        save_top_k=-1,
        every_n_epochs=hypes['train_params']['eval_freq'],  # 每几个周期后保存检查点
        verbose=True
    )
    from lightning.pytorch.callbacks import LearningRateMonitor
    class CustomLearningRateMonitor(LearningRateMonitor):
        def on_train_epoch_end(self, trainer, pl_module):
            super().on_train_epoch_end(trainer, pl_module)
            print()
            for name, lr in self.lrs.items():
                print(f"Epoch {trainer.current_epoch}: {name}'s Learning rate: {lr[-1]}")

    lr_monitor = CustomLearningRateMonitor(logging_interval='epoch')


    print('-----------------Seed Setting----------------------')
    if 'reproduce_the_experiment' not in hypes:
        seed = cmt_utils.init_random_seed(None if opt.seed == 0 else opt.seed)
        hypes['train_params']['seed'] = seed
        seed_everything(seed)
        logging.basicConfig(filename=os.path.dirname(saved_path) + '/seed.log', level=logging.INFO)
        logging.info(saved_path + ' Set seed to %d' % seed)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=6,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=6,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    model = MyModel(hypes)

    if opt.load_cmt_camera_nofusion_weights:
        model = load_cmt_camera_nofusion_checkpoint(model, opt.cmt_camera_nofusion_weights_path)
    if opt.load_cmt_lidar_nofusion_weights:
        model = load_cmt_lidar_nofusion_checkpoint(model, opt.cmt_lidar_nofusion_weights_path)
    if opt.ckpt_path:
        model = load_ckpt(model, opt.ckpt_path)

    # Set freeze flag for camera/lidar feature extractors
    if opt.fix_camera_backbone:
        model.model.fix_camera_backbone()

    if opt.fix_lidar_backbone:
        model.model.fix_lidar_backbone()

    trainer = L.Trainer(max_epochs=hypes['train_params']['epoches'],
                        precision="16-mixed", accelerator="gpu",
                        devices=1,
                        check_val_every_n_epoch=hypes['train_params']['eval_freq'],
                        log_every_n_steps=50,
                        default_root_dir=saved_path,
                        gradient_clip_val=35, gradient_clip_algorithm="norm",
                        callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
