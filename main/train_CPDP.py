import argparse
import os

import numpy as np
import pandas

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from utils import get_lr_scheduler, set_optimizer_lr, get_dataloader
from tqdm import tqdm
from resnet import ResNet18, ResNet50, ResNet101, ResNet152
from sklearn.metrics import precision_recall_fscore_support, roc_curve, roc_auc_score

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, num_epoch, train_iter, valid_iter, loss, optimizer, lr_adjust_function, target_size, src, tgt,
                device):
    data_transforms = transforms.Compose([
        transforms.Resize(target_size[0]),
        transforms.CenterCrop(target_size[1]),
        transforms.ToTensor()
    ])

    train_best_f1 = 0
    valid_best_f1 = 0
    valid_best_f1_all = 0
    fpr_best, tpr_best, thresholds_best = [], [], []
    auc_score_best = 0

    for current_epoch in range(num_epoch):
        set_optimizer_lr(optimizer, lr_adjust_function, current_epoch)

        predicts = []
        trues = []

        pbar = tqdm(total=epoch_step, desc=f'{src}->{tgt}_train:Epoch {current_epoch + 1}/{epoch}', postfix=dict,
                    mininterval=0.3)
        model.train()

        for iteration, batch in enumerate(train_iter):
            images, labels = batch[0], batch[1]
            # 创建一个空的张量来存储合成的图像
            num_images = len(batch[0])
            train_tensor = torch.zeros((num_images, 3, target_size[1], target_size[1]), dtype=torch.float32)

            with torch.no_grad():
                for i, data in enumerate(images):
                    image = Image.open(data)
                    image = data_transforms(image)
                    image = image.float() / 255.0  # 转换为张量并归一化
                    train_tensor[i] = image

                train_tensor = train_tensor.to(device)
                labels = labels.to(device)

            # Clear Gradient
            optimizer.zero_grad()

            outputs = model(train_tensor)
            output = loss(outputs, labels)

            # 反向传播
            output.backward()
            optimizer.step()

            with torch.no_grad():
                _, max_indices = torch.max(nn.Sigmoid()(outputs), dim=1)
                equal = torch.eq(max_indices, labels)
                accuracy = torch.mean(equal.float())
                prep = max_indices.cpu().numpy()
                label = labels.cpu().numpy()

                p, r, f, _ = precision_recall_fscore_support(label, prep, average='weighted', zero_division=0)

            if f > train_best_f1:
                train_best_f1 = f

            pbar.set_postfix(**{'loss': output.item(),
                                'acc': accuracy.item(),
                                'precision': p,
                                'recall': r,
                                "f1-score": f,
                                "train_best_f1": train_best_f1,
                                'lr': get_lr(optimizer)})
            pbar.update(1)
        pbar.close()

        # Validating
        pbar2 = tqdm(total=epoch_step_val, desc=f'{src}->{tgt}_valid:Epoch {current_epoch + 1}/{epoch}', postfix=dict,
                     mininterval=0.3)

        model.eval()
        for iteration, batch in enumerate(valid_iter):
            images, labels = batch[0], batch[1]
            # 创建一个空的张量来存储合成的图像
            num_images = len(batch[0])
            valid_tensor = torch.zeros((num_images, 3, target_size[1], target_size[1]), dtype=torch.float32)

            with torch.no_grad():
                for i, data in enumerate(images):
                    image = Image.open(data)
                    image = data_transforms(image)
                    image = image.float() / 255.0  # 转换为张量并归一化
                    valid_tensor[i] = image

                valid_tensor = valid_tensor.to(device)
                labels = labels.to(device)

                # Clear Gradient
                optimizer.zero_grad()

                outputs = model(valid_tensor)
                output = loss(outputs, labels)

                _, max_indices = torch.max(nn.Sigmoid()(outputs), dim=1)
                equal = torch.eq(max_indices, labels)
                accuracy = torch.mean(equal.float())

                _, max_indices = torch.max(nn.Sigmoid()(outputs), dim=1)
                prep = max_indices.cpu().numpy()

                label = labels.cpu().numpy()
                p, r, f, _ = precision_recall_fscore_support(label, prep, average='weighted', zero_division=0)
                predicts.extend(prep)
                trues.extend(label)

            if f > valid_best_f1:
                valid_best_f1 = f

            pbar2.set_postfix(**{'val_loss': output.item(),
                                 'acc': accuracy.item(),
                                 'precision': p,
                                 'recall': r,
                                 "f1-score": f,
                                 "valid_best_f1": valid_best_f1})
            pbar2.update(1)
        pbar2.close()

        precision, recall, f_1, _ = precision_recall_fscore_support(trues, predicts, average='weighted', zero_division=0)
        fpr, tpr, thresholds = roc_curve(trues, predicts)
        auc_score = roc_auc_score(trues, predicts)
        if f_1 > valid_best_f1_all:
            valid_best_f1_all = f_1
            fpr_best = fpr
            tpr_best = tpr
            thresholds_best = thresholds
            auc_score_best = auc_score

        # print("Total valid results(P,R,F1): %.4f, %.4f, %.4f" % (precision, recall, f1))

    print('Finish')
    return valid_best_f1_all, fpr_best, tpr_best, thresholds_best, auc_score_best


def determine_batch_size(sources, targets):
    with open(sources, 'r') as f, open(targets, 'r') as f2:
        sources_data = f.readlines()
        targets_data = f2.readlines()

        sources_len = len(sources_data)
        targets_len = len(targets_data)

        min_len = sources_len if sources_len < targets_len else targets_len

        # 根据数据集大小确定 batch_size
        if min_len > 500:
            batch_size = 32
        elif min_len > 200:
            batch_size = 16
        elif min_len > 100:
            batch_size = 8
        elif min_len > 50:
            batch_size = 4
        else:
            batch_size = 2

    return batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path', type=str, default="../data/img_tsbt_4w", help='数据集地址')
    parser.add_argument('--save_path', type=str, default='result_CPDP_tsbt_4w.xlsx', help='保存结果文件')
    args = parser.parse_args()
    save_path = args.save_path

    target_size = (230, 224)  # 设置目标图像的大小(0位置为缩放后大小，1位置为随机剪裁后大小)
    channels = 3

    # config
    config = {
        "use_gpu": True,
        "num_epoch": 100,
        "init_epoch": 0,
        "optimizer_type": "adam",
        "input_shape": target_size,
        "dataset_path": args.dataset_path,
        "init_lr": 1e-4,
        "min_lr": 1e-5,
        "momentum": 0.9,
        "lr_decay_type": 'cos',
        "num_workers": 5,
        "weight_decay": 5,
        "net": "resnet",
        "run_times": 10
    }

    optimizer_type = config['optimizer_type']
    init_shape = config['input_shape']
    epoch = config['num_epoch']
    init_epoch = config['init_epoch']
    dataset_path = config['dataset_path']
    init_lr = config['init_lr']
    min_lr = config['min_lr']
    lr_decay_type = config['lr_decay_type']
    momentum = config['momentum']
    num_workers = config['num_workers']
    weight_decay = config["weight_decay"]
    network = config['net']
    run_times = config['run_times']

    # CPDP
    sources = ['ant-1.6', 'jedit-4.1', 'camel-1.4', 'poi-3.0', 'camel-1.4', 'log4j-1.1', 'jedit-4.1', 'lucene-2.2',
               'lucene-2.2', 'xerces-1.3', 'xalan-2.5', 'log4j-1.1', 'xalan-2.5', 'ivy-2.0', 'xerces-1.3',
               'synapse-1.2',
               'ivy-1.4', 'poi-2.5.1', 'ivy-2.0', 'poi-3.0', 'synapse-1.2', 'ant-1.6']
    targets = ['camel-1.4', 'camel-1.4', 'ant-1.6', 'ant-1.6', 'jedit-4.1', 'jedit-4.1', 'log4j-1.1', 'log4j-1.1',
               'xalan-2.5',
               'xalan-2.5', 'lucene-2.2', 'lucene-2.2', 'xerces-1.3', 'xerces-1.3', 'ivy-2.0', 'ivy-2.0', 'synapse-1.1',
               'synapse-1.1',
               'synapse-1.2', 'synapse-1.2', 'poi-3.0', 'poi-3.0']

    # WPDP
    # sources = ['ant-1.5', 'ant-1.6', 'camel-1.2', 'camel-1.4', 'jedit-3.2.1', 'jedit-4.0', 'log4j-1.0', 'lucene-2.0',
    #            'lucene-2.2', 'xalan-2.4', 'xerces-1.2', 'ivy-1.4', 'synapse-1.0', 'synapse-1.1', 'poi-1.5', 'poi-2.5.1']
    # targets = ['ant-1.6', 'ant-1.7', 'camel-1.4', 'camel-1.6', 'jedit-4.0', 'jedit-4.1', 'log4j-1.1', 'lucene-2.2',
    #            'lucene-2.4', 'xalan-2.5', 'xerces-1.3', 'ivy-2.0', 'synapse-1.1', 'synapse-1.2', 'poi-2.5.1', 'poi-3.0']

    # ALL
    #    sources = ['train_instance']
    #    targets = ['test_instance']

    result = pandas.DataFrame(columns=['Experiment', 'F1_avg', 'fpr', 'tpr', 'thresholds', 'auc'])

    for i in tqdm(range(0, len(sources))):
        # Dataset
        source = os.path.join(dataset_path + '/' + sources[i], sources[i] + '.csv')
        target = os.path.join(dataset_path + '/' + targets[i], targets[i] + '.csv')

        batch_size = determine_batch_size(source, target)

        total_F1 = 0
        total_fpr = np.array([])
        total_tpr = np.array([])
        total_thresholds = np.array([])
        total_auc = 0
        for _ in tqdm(range(run_times)):

            # Device, models... Loss
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("device:" + str(device))

            model = ResNet18(2, channels=channels)
            # model = ResNet50(2)

            num_parameters = count_parameters(model)

            print(f"Num parameters:{num_parameters}")

            loss = nn.CrossEntropyLoss()

            if config['use_gpu']:
                model.cuda()
            # Dataloader & dataset
            print("Getting dataloader...")
            train_iter, valid_iter = get_dataloader(source, target, batch_size, num_workers)

            num_train = len(train_iter) * batch_size
            num_val = len(valid_iter) * batch_size

            nbs = 64
            lr_limit_max = 1e-5 if optimizer_type == 'adam' else 1e-5
            lr_limit_min = 3e-6 if optimizer_type == 'adam' else 5e-5
            Init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            optimizer = {
                'adam': torch.optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.9),
                                         weight_decay=weight_decay),
                'sgd': torch.optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                                       weight_decay=weight_decay)
            }[optimizer_type]

            lr_adjust_function = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epoch)

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("The dataset is too small, please expand the data set.")


            def init_weights(m):
                if type(m) == nn.Linear or type(m) == nn.Conv2d:
                    nn.init.xavier_uniform_(m.weight)


            # Train

            print("Init Model---------------------------------------------")
            model.apply(init_weights)
            f1, fpr_, tpr_, thresholds_, auc_score_ = train_model(model, epoch, train_iter, valid_iter, loss, optimizer,
                                                                  lr_adjust_function, target_size,
                                                                  sources[i], targets[i], device)

            total_F1 += f1
            total_auc += auc_score_

            total_fpr = np.array(fpr_)
            total_tpr = np.array(tpr_)
            total_thresholds = np.array(thresholds_)

        row = {'Experiment': sources[i] + '->' + targets[i], 'F1_avg': total_F1 / run_times, 'fpr': total_fpr,
               'tpr': total_tpr, 'thresholds': total_thresholds, 'auc': total_auc / run_times}
        result = result.append(row, ignore_index=True)
        result.to_excel(save_path)

