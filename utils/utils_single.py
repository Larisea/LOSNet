import sys
from pathlib import Path
import time
import numpy as np
import torch.nn.functional as F
from utils.path_hyperparameter import ph
import torch
import logging
from tqdm import tqdm
import wandb
from PIL import Image


def save_model(model, path, epoch, mode, optimizer=None):
    # mode should be checkpoint or loss or f1score
    Path(path).mkdir(parents=True,
                     exist_ok=True)  # create a dictionary
    # ipdb.set_trace()
    # 替换不合法空格冒号
    localtime = time.asctime(time.localtime(time.time())).replace(" ", "_").replace(":", "_")
    if mode == 'checkpoint':
        state_dict = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state_dict,
                   str(path + f'checkpoint_epoch{epoch}_{localtime}.pth'))
    else:
        torch.save(model.state_dict(), str(path + f'best_{mode}_epoch{epoch}_{localtime}.pth'))
    logging.info(f'best {mode} model {epoch} saved at {localtime}!')


# parameter [warmup_lr, grad_scaler] is required in training
# parameter [best_metrics, checkpoint_path, best_f1score_model_path, best_loss_model_path, non_improved_epoch]
# is required in evaluation
def train_val(
        mode, dataset_name,
        dataloader, device, log_wandb, net, optimizer, total_step,
        lr, criterion, metric_collection, to_pilimg, epoch,
        warmup_lr=None, grad_scaler=None,
        best_metrics=None, checkpoint_path=None,
        best_f1score_model_path=None, best_loss_model_path=None, non_improved_epoch=None
):
    assert mode in ['train', 'val'], 'mode should be train or val'
    epoch_loss = 0
    dataset_list = ['A', 'B', 'OUT']
    # Begin Training/Evaluating/Testing
    if mode == 'train':
        net.train()
    else:
        net.eval()
    logging.info(f'SET model mode to {mode}!')
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
        if mode == 'train':
            # using amp
            with torch.cuda.amp.autocast():
                # preds = net(batch_img1, batch_img2)
                preds, mask1, mask2, coarse = net(batch_img1, batch_img2)
                loss1 = criterion(preds, labels)
                loss2 = criterion(coarse, labels)
                loss3 = criterion(mask1, labels)
                loss4 = criterion(mask2, labels)
                # loss5 = criterion(mask3, labels)
                loss = loss1 + loss2 + loss3 + loss4
            cd_loss = sum(loss)
            grad_scaler.scale(cd_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 20, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            preds, mask1, mask2, coarse = net(batch_img1, batch_img2)
            loss1 = criterion(preds, labels)
            loss2 = criterion(coarse, labels)
            loss3 = criterion(mask1, labels)
            loss4 = criterion(mask2, labels)
            loss = loss1 + loss2 + loss3 + loss4
            cd_loss = sum(loss)
        epoch_loss += cd_loss
        # 最终预测图
        preds = torch.sigmoid(preds)

        # log the t1_img, t2_img, pred and label
        if i == sample_batch:
            sample_index = np.random.randint(low=0, high=batch_img1.shape[0])
            # ipdb.set_trace()
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
    # 将指标保存到文本文件中
    with open("metrics.txt", "a") as file:
        file.write(f"{mode}-epoch {epoch + 1} metrics:\n")
        file.write(formatted_output + "\n")

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
                print('save best model!')
                save_model(net, best_f1score_model_path, epoch, 'f1score')
        elif epoch_loss < best_metrics['lowest loss']:
            best_metrics['lowest loss'] = epoch_loss
            if ph.save_best_model:
                print('save best model!')
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
