import os
import math

import pandas
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.data.dataset import Dataset


# Use to create dataloader for training.

class CloneDetectionDataset(Dataset):

    # Dataloader的目标:载入数据、提供数据

    def __init__(self, code_imgs, labels):
        self.code_imgs = code_imgs
        self.labels = labels

    def __len__(self):
        return len(self.code_imgs)

    def __getitem__(self, item):
        return self.code_imgs[item], self.labels[item]


def get_dataloader(source_file, target_file, batch_size, num_workers):
    # To load dataset from a file and return the train dataset and valid dataset

    # 获取pair信息和训练信息
    sources = pandas.read_csv(source_file)
    targets = pandas.read_csv(target_file)

    train_path = []
    train_label = []
    valid_path = []
    valid_label = []

    print("processing...")
    for _, row in sources.iterrows():
        path = row['Path']
        label = row['Label']
        train_path.append(path)
        train_label.append(label)
    for _, row in targets.iterrows():
        path = row['Path']
        label = row['Label']
        valid_path.append(path)
        valid_label.append(label)

    print("Building dataset...")
    train_dataset = CloneDetectionDataset(train_path, train_label)
    valid_dataset = CloneDetectionDataset(valid_path, valid_label)

    shuffle = True

    print("Building dataloader...")
    train_dataloader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    return train_dataloader, valid_dataloader


def set_optimizer_lr(optimizer, lr_adjust_function, epoch):
    """
    optimizer.param_groups是一个包含字典的列表，每个字典表示优化器中的一个参数组。在PyTorch中，一个模型的参数可以按照不同的方式进行分组，并为每个参数组设置不同的学习率、权重衰减等超参数。

每个参数组的字典包含以下常见键值对：

    'params'：一个包含了该参数组中要优化的参数的列表。
    'lr'：该参数组的学习率（learning rate）。
    'weight_decay'：该参数组的权重衰减（weight decay）。
    'momentum'：该参数组的动量（momentum）。
    'dampening'：该参数组的阻尼（dampening）。
    'nesterov'：该参数组是否使用Nesterov动量（nesterov momentum）。

    """

    lr = lr_adjust_function(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    """
    过于复杂的算法，暂时不看
    """

    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
                                              ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0
                    + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iter)
            )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func
