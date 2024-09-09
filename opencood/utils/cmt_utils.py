import torch

def load_cmt_camera_nofusion_checkpoint(model, weight_path):
    # 加载预训练权重
    pretrained_weights = torch.load(weight_path)

    # 创建一个新的字典来存储修改后的键和值
    modified_state_dict = {}

    # 遍历 state_dict 并修改键
    for key in pretrained_weights['state_dict']:
        if key.startswith('model.cmt_c'):
            # 修改键
            modified_key = key.replace("model.cmt_c", "model.camera_encoder.cmt_c")

            # 获取原始值
            value = pretrained_weights['state_dict'][key]

            # 将修改后的键和原始值存储到新的字典中
            modified_state_dict[modified_key] = value
        elif key.startswith('model.camera_task_heads'):
            # 修改键
            modified_key = key.replace("model.camera_task_heads", "model.camera_encoder.camera_task_heads")

            # 获取原始值
            value = pretrained_weights['state_dict'][key]

            # 将修改后的键和原始值存储到新的字典中
            modified_state_dict[modified_key] = value
        else:
            # 想要忽略的键值对
            continue

    # 使用修改后的 state_dict
    model.load_state_dict(modified_state_dict, strict=False)

    # 检查是否导入成功
    after_load_model = {k: v for k, v in model.state_dict().items() if k.startswith('model.camera_encoder')}
    origin_model = {k: v.to(torch.device('cpu')) for k, v in modified_state_dict.items() if k.startswith('model.camera_encoder')}
    is_equal = after_load_model.keys() == origin_model.keys() and all(
        torch.equal(after_load_model[key], origin_model[key]) for key in after_load_model.keys())
    del after_load_model
    del origin_model
    if is_equal:
        print('load cmt camera nofusion checkpoint success!')
    return model

def load_cmt_lidar_nofusion_checkpoint(model, weight_path):
    # 加载预训练权重
    pretrained_weights = torch.load(weight_path)

    # 创建一个新的字典来存储修改后的键和值
    modified_state_dict = {}

    # 遍历 state_dict 并修改键
    for key in pretrained_weights['state_dict']:
        if key.startswith('model.cmt_l'):
            # 修改键
            modified_key = key.replace("model.cmt_l", "model.lidar_encoder.cmt_l")

            # 获取原始值
            value = pretrained_weights['state_dict'][key]

            # 将修改后的键和原始值存储到新的字典中
            modified_state_dict[modified_key] = value
        elif key.startswith('model.lidar_task_heads'):
            # 修改键
            modified_key = key.replace("model.lidar_task_heads", "model.lidar_encoder.lidar_task_heads")

            # 获取原始值
            value = pretrained_weights['state_dict'][key]

            # 将修改后的键和原始值存储到新的字典中
            modified_state_dict[modified_key] = value
        else:
            # 想要忽略的键值对
            continue

    # 使用修改后的 state_dict
    model.load_state_dict(modified_state_dict, strict=False)

    # 检查是否导入成功
    after_load_model = {k: v for k, v in model.state_dict().items() if k.startswith('model.lidar_encoder')}
    origin_model = {k: v.to(torch.device('cpu')) for k, v in modified_state_dict.items() if k.startswith('model.lidar_encoder')}
    is_equal = after_load_model.keys() == origin_model.keys() and all(
        torch.equal(after_load_model[key], origin_model[key]) for key in after_load_model.keys())
    del after_load_model
    del origin_model
    if is_equal:
        print('load cmt lidar nofusion checkpoint success!')
    return model

def load_ckpt(model, weight_path):
    pretrained_weights = torch.load(weight_path)
    model.load_state_dict(pretrained_weights['state_dict'], strict=False)
    return model

import torch.distributed as dist
import numpy as np
from opencood.tools.multi_gpu_utils import get_dist_info
def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

import os
import re
import glob
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch_ = max(epochs_exist)
    else:
        initial_epoch_ = 0
    return initial_epoch_


def load_saved_model(saved_path, model, epoch=-1):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.
    epoch: int
        which epoch to load

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    if os.path.exists(os.path.join(saved_path, 'epoch=%d.ckpt' % epoch)):
        initial_epoch = epoch
        checkpoint = torch.load(os.path.join(saved_path, 'epoch=%d.ckpt' % epoch),
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
    else:
        if epoch > 0:
            initial_epoch = epoch
        else:
            initial_epoch = findLastCheckpoint(saved_path)

        if initial_epoch > 0:
            print('resuming by loading epoch %d' % initial_epoch)
            checkpoint = torch.load(
                os.path.join(saved_path,
                             'net_epoch%d.pth' % initial_epoch),
                map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)

            del checkpoint

    return initial_epoch, model