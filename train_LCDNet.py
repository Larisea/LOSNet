import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, JaccardIndex
from tqdm import tqdm

from LCDNet.LCDNetV3Pro import LCDNetV3Pro
from LCDNet.comparion.FC_Siam_conc import SiamUnet_conc
# from LCDNet.Zoo.LCDNet import LCDNet
from loss.change_loss import change_loss
from utils.data_loading import BasicDataset
from utils.path_hyperparameter import ph


class DataLoaderX(DataLoader):
    """Using prefetch_generator to accelerate data loading

    原本 PyTorch 默认的 DataLoader 会创建一些 worker 线程来预读取新的数据，但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据。
    使用 prefetch_generator，我们可以保证线程不会等待，每个线程都总有至少一个数据在加载。

    Parameter:
        DataLoader(class): torch.utils.data.DataLoader.
    """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True


def get_dataset_stats(data_name):
    """
    根据数据集路径或名字返回对应的均值和标准差参数。
    Args:
        data_name (str): 数据集路径或名字，例如 "./CDData/SYSU-CD/" 或 "SYSU-CD"

    Returns:
        tuple: (t1_mean, t1_std, t2_mean, t2_std) 包含两个时间点的均值和标准差
    """
    # 提取最后一部分作为数据集名字
    dataset = data_name.strip("/").split("/")[-1]
    # 根据数据集名字返回对应的参数，并提示用户
    if dataset == "LEVIRCD":
        print(f"正在使用数据集: {dataset}")
        t1_mean, t1_std = [0.45026044, 0.44666811, 0.38134658], [0.17456748, 0.16490024, 0.15318057]
        t2_mean, t2_std = [0.34552285, 0.33819558, 0.28881546], [0.12937804, 0.12601846, 0.1187869]
    elif dataset == "WHU-CD":
        print(f"正在使用数据集: {dataset}")
        t1_mean, t1_std = [0.48435662, 0.44378854, 0.38627573], [0.14271143, 0.13803999, 0.13800791]
        t2_mean, t2_std = [0.48289366, 0.48230296, 0.45964595], [0.17929289, 0.16960935, 0.17576571]
    elif dataset == "SYSU-CD":
        print(f"正在使用数据集: {dataset}")
        t1_mean, t1_std = [0.39659575, 0.52846196, 0.46540029], [0.20213537, 0.15811189, 0.15296703]
        t2_mean, t2_std = [0.40202364, 0.48766127, 0.39895688], [0.18235275, 0.15682769, 0.1543715]
    elif dataset == "CDD":
        print(f"正在使用数据集: {dataset}")
        t1_mean, t1_std = [0.3538937 , 0.39104213, 0.34306571], [0.1533363 , 0.15479208, 0.1429063]
        t2_mean, t2_std = [0.4732442 , 0.49861559, 0.46873127], [0.16068315, 0.16395893, 0.15758859]
    else:
        raise ValueError(f"未知的数据集: {dataset}，请检查输入是否正确")

    return t1_mean, t1_std, t2_mean, t2_std


def train_net(dataset_name):
    # 1. Create dataset

    dataset_list = ['A', 'B', 'OUT']
    # dataset_list = ['t1', 't2', 'label']
    # 1. Create dataset, checkpoint and best model path
    # compute mean and std of train dataset to normalize train/val dataset
    # t1_mean, t1_std = compute_mean_std(images_dir=f'{dataset_name}/train/{dataset_list[0]}/')
    # t2_mean, t2_std = compute_mean_std(images_dir=f'{dataset_name}/train/{dataset_list[1]}/')
    # 获取提前计算好的方差和标准差节约计算时间
    t1_mean, t1_std, t2_mean, t2_std = get_dataset_stats(dataset_name)
    # dataset path should be dataset_name/train or val/t1 or t2 or label
    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)
    train_dataset = BasicDataset(t1_images_dir=f'{dataset_name}/train/{dataset_list[0]}/',
                                 t2_images_dir=f'{dataset_name}/train/{dataset_list[1]}/',
                                 labels_dir=f'{dataset_name}/train/{dataset_list[2]}/',
                                 train=True, **dataset_args)
    val_dataset = BasicDataset(t1_images_dir=f'{dataset_name}/val/{dataset_list[0]}/',
                               t2_images_dir=f'{dataset_name}/val/{dataset_list[1]}/',
                               labels_dir=f'{dataset_name}/val/{dataset_list[2]}/',
                               train=False, **dataset_args)

    # 2. Markdown dataset size
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. Create data loaders
    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True
                       )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=ph.batch_size, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False,
                            batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 4. Initialize logging

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # working device
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime
    # using wandb to log hyperparameter, metrics and output
    # resume=allow means if the id is identical with the previous one, the run will resume
    # (anonymous=must) means the id will be anonymous
    log_wandb = wandb.init(project=ph.log_wandb_project,
                           resume='allow',
                           anonymous='must',
                           settings=wandb.Settings(start_method='thread'),
                           config=hyperparameter_dict)
    logging.info(f'''Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
        Mixed Precision: {ph.amp}
    ''')

    # 5. Set up model, optimizer, warm_up_scheduler, learning rate scheduler, loss function and other things

    net = LCDNet()  # change detection model
    # 提取模型的类名
    net = net.to(device=device)
    optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate,
                            weight_decay=ph.weight_decay)  # optimizer
    warmup_lr = np.arange(1e-7, ph.learning_rate,
                          (ph.learning_rate - 1e-7) / ph.warm_up_step)  # warm up learning rate
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=ph.patience,
    #                                                  factor=ph.factor)  # learning rate scheduler
    # grad_scaler = torch.cuda.amp.GradScaler()  # loss scaling for amp
    grad_scaler = torch.amp.GradScaler(device='cuda')  # loss scaling for amp

    # load model and optimizer
    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint['net'])
        logging.info(f'Model loaded from {ph.load}')
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = ph.learning_rate
            optimizer.param_groups[0]['capturable'] = True

    total_step = 0  # logging step
    lr = ph.learning_rate  # learning rate

    # criterion = FCCDN_loss_without_seg  # loss function
    criterion = change_loss  # loss function

    best_metrics = dict.fromkeys(['best_f1score', 'lowest loss'], 0)  # best evaluation metrics
    metric_collection = MetricCollection({
        'accuracy': Accuracy(task="binary").to(device=device),
        'precision': Precision(task="binary").to(device=device),
        'recall': Recall(task="binary").to(device=device),
        'f1score': F1Score(task="binary").to(device=device),
        # 增加IOU
        'Iou': JaccardIndex(task="binary").to(device=device)
    })  # metrics calculator

    to_pilimg = T.ToPILImage()  # convert to PIL image to log in wandb

    # model saved path
    checkpoint_path = f'{dataset_name}_checkpoint/'
    best_f1score_model_path = f'{dataset_name}_best_f1score_model/'
    best_loss_model_path = f'{dataset_name}_best_loss_model/'

    non_improved_epoch = 0  # adjust learning rate when non_improved_epoch equal to patience

    # 5. Begin training

    for epoch in range(ph.epochs):
        log_wandb, net, optimizer, grad_scaler, total_step, lr = \
            train_val(
                mode='train', dataset_name=dataset_name, dataset_list=dataset_list,
                dataloader=train_loader, device=device, log_wandb=log_wandb, net=net,
                optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                warmup_lr=warmup_lr, grad_scaler=grad_scaler
            )

        # 6. Begin evaluation

        # starting validation from evaluate epoch to minimize time
        if epoch >= ph.evaluate_epoch:
            with torch.no_grad():
                log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch = \
                    train_val(
                        mode='val', dataset_name=dataset_name, dataset_list=dataset_list,
                        dataloader=val_loader, device=device, log_wandb=log_wandb, net=net,
                        optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                        metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                        best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                        best_f1score_model_path=best_f1score_model_path, best_loss_model_path=best_loss_model_path,
                        non_improved_epoch=non_improved_epoch
                    )

    wandb.finish()
    # os.system('shutdown')


def save_model(model, path, epoch, mode, optimizer=None):
    # mode should be checkpoint or loss or f1score
    Path(path).mkdir(parents=True,
                     exist_ok=True)  # create a dictionary
    # ipdb.set_trace()
    net_name = model.__class__.__name__
    # 替换不合法空格冒号
    localtime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if mode == 'checkpoint':
        state_dict = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state_dict, str(path + f'{localtime}_Epoch_{epoch}_checkpoint_{net_name}.pth'))
    else:
        torch.save(model.state_dict(), str(path + f'{localtime}_Epoch_{epoch}_best_{mode}_{net_name}.pth'))
    logging.info(f'best {mode} LCDNet {epoch} saved at {localtime}!')


# parameter [warmup_lr, grad_scaler] is required in training
# parameter [best_metrics, checkpoint_path, best_f1score_model_path, best_loss_model_path, non_improved_epoch]
# is required in evaluation
def train_val(
        mode, dataset_name, dataset_list,
        dataloader, device, log_wandb, net, optimizer, total_step,
        lr, criterion, metric_collection, to_pilimg, epoch,
        warmup_lr=None, grad_scaler=None,
        best_metrics=None, checkpoint_path=None,
        best_f1score_model_path=None, best_loss_model_path=None, non_improved_epoch=None
):
    assert mode in ['train', 'val'], 'mode should be train or val'
    epoch_loss = 0
    # Begin Training/Evaluating/Testing
    if mode == 'train':
        net.train()
    else:
        net.eval()
    logging.info(f'SET mode to {mode}!')
    batch_iter = 0

    tbar = tqdm(dataloader)
    n_iter = len(dataloader)
    sample_batch = np.random.randint(low=0, high=n_iter)

    for i, (batch_img1, batch_img2, labels, name) in enumerate(tbar):
        tbar.set_description(
            "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + ph.batch_size))
        batch_iter = batch_iter + ph.batch_size
        total_step += 1

        # Zero the gradient if train
        if mode == 'train':
            optimizer.zero_grad()
            # warm up
            if total_step < ph.warm_up_step:
                for g in optimizer.param_groups:
                    g['lr'] = warmup_lr[total_step]

        batch_img1 = batch_img1.float().to(device)
        batch_img2 = batch_img2.float().to(device)
        labels = labels.float().to(device)
        seg_label = F.interpolate(labels.unsqueeze(1), scale_factor=1 / 2, mode='bilinear').squeeze(1).to(device)
        if mode == 'train':
            # using amp
            with torch.cuda.amp.autocast():
                preds = net(batch_img1, batch_img2)
                loss = criterion(preds, labels)
            cd_loss = sum(loss)
            grad_scaler.scale(cd_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 20, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            preds = net(batch_img1, batch_img2)
            loss = criterion(preds, labels)
            cd_loss = sum(loss)
        epoch_loss += cd_loss
        # 最终预测图
        preds = torch.sigmoid(preds)

        # log the t1_img, t2_img, pred and label
        if i == sample_batch:
            sample_index = np.random.randint(low=0, high=batch_img1.shape[0])
            t1_images_dir = Path(f'{dataset_name}/{mode}/{dataset_list[0]}/')
            t2_images_dir = Path(f'{dataset_name}/{mode}/{dataset_list[1]}/')
            labels_dir = Path(f'{dataset_name}/{mode}/{dataset_list[2]}/')
            t1_img_log = Image.open(list(t1_images_dir.glob(name[sample_index] + '.*'))[0])
            t2_img_log = Image.open(list(t2_images_dir.glob(name[sample_index] + '.*'))[0])
            label_log = Image.open(list(labels_dir.glob(name[sample_index] + '.*'))[0])
            pred_log = preds[sample_index].cpu().clone()
            pred_log[pred_log >= 0.5] = 1
            pred_log[pred_log < 0.5] = 0
            pred_log = pred_log.float()

        preds = preds.float()
        labels = labels.int().unsqueeze(1)
        batch_metrics = metric_collection.forward(preds, labels)  # compute metric

        # log loss and metric
        log_wandb.log({
            f'{mode} loss': cd_loss,
            f'{mode} accuracy': batch_metrics['accuracy'],
            f'{mode} precision': batch_metrics['precision'],
            f'{mode} recall': batch_metrics['recall'],
            f'{mode} f1score': batch_metrics['f1score'],
            f'{mode} Iou': batch_metrics['Iou'],
            'learning rate': optimizer.param_groups[0]['lr'],
            f'{mode} loss_dice': loss[0],
            f'{mode} loss_bce': loss[1],
            'step': total_step,
            'epoch': epoch
        })

        # clear batch variables from memory
        del batch_img1, batch_img2, labels, name

    epoch_metrics = metric_collection.compute()  # compute epoch metric
    # 格式化输出指标
    metrics_output = epoch_metrics
    formatted_output = "Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f},IOU: {:.4f}, Accuracy: {:.4f} ".format(
        metrics_output['precision'].item() * 100,
        metrics_output['recall'].item() * 100,
        metrics_output['f1score'].item() * 100,
        metrics_output['Iou'].item() * 100,
        metrics_output['accuracy'].item() * 100
    )
    print(f"{mode}-epoch {epoch + 1} metrics is:{formatted_output}")

    epoch_loss /= n_iter
    for k in epoch_metrics.keys():
        log_wandb.log({f'epoch_{mode}_{str(k)}': epoch_metrics[k],
                       'epoch': epoch})  # log epoch metric
    metric_collection.reset()
    log_wandb.log({f'epoch_{mode}_loss': epoch_loss,
                   'epoch': epoch})  # log epoch loss

    log_wandb.log({
        f'{mode} t1_images': wandb.Image(t1_img_log),
        f'{mode} t2_images': wandb.Image(t2_img_log),
        f'{mode} masks': {
            'label': wandb.Image(label_log),
            'pred': wandb.Image(to_pilimg(pred_log)),
        },
        'epoch': epoch
    })  # log the t1_img, t2_img, pred and label

    # save best model and adjust learning rate according to learning rate scheduler
    if mode == 'val':
        if epoch_metrics['f1score'] > best_metrics['best_f1score']:
            non_improved_epoch = 0
            best_metrics['best_f1score'] = epoch_metrics['f1score']
            if ph.save_best_model:
                print('save best_f1score model!')
                save_model(net, best_f1score_model_path, epoch, 'f1score')
        elif epoch_loss < best_metrics['lowest loss']:
            best_metrics['lowest loss'] = epoch_loss
            if ph.save_best_model:
                print('save lowest loss model!')
                save_model(net, best_loss_model_path, epoch, 'loss')
        else:
            non_improved_epoch += 1
            if non_improved_epoch == ph.patience:
                lr *= ph.factor
                for g in optimizer.param_groups:
                    g['lr'] = lr
                non_improved_epoch = 0

        # save checkpoint every specified interval
        if (epoch + 1) % ph.save_interval == 0 and ph.save_checkpoint:
            save_model(net, checkpoint_path, epoch, 'checkpoint', optimizer=optimizer)
    print(f"{mode}-epoch {epoch + 1} loss and lr is:{epoch_loss},{lr}")
    if mode == 'train':
        return log_wandb, net, optimizer, grad_scaler, total_step, lr
    elif mode == 'val':
        return log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch
    else:
        print('Error')
        sys.exit(0)


if __name__ == '__main__':

    # set random seed to make the experiment reproducible
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=f'{ph.dataset_name}')
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)
