import time
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
# import logging
from tqdm import tqdm

from utils.path_hyperparameter import ph


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


def train_val(
        mode, dataloader, device, net, optimizer, total_step,
        lr, criterion, metric_collection, epoch,
        warmup_lr=None, grad_scaler=None,
        best_metrics=None, checkpoint_path=None,
        best_f1score_model_path=None, best_loss_model_path=None, non_improved_epoch=None):
    assert mode in ['train', 'val'], 'mode should be train, val'
    epoch_loss = 0
    # Begin Training/Evaluating
    if mode == 'train':
        net.train()
    else:
        net.eval()
    batch_iter = 0

    tbar = tqdm(dataloader)
    n_iter = len(dataloader)
    np.random.randint(low=0, high=n_iter)

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
        # seg_label = F.interpolate(labels.unsqueeze(1), scale_factor=1 / 2, mode='bilinear').squeeze(1).to(device)

        if mode == 'train':
            # using amp
            with torch.cuda.amp.autocast():
                preds, mask, m1, m2, m3, m4 = net(batch_img1, batch_img2)
                loss1 = criterion(preds, labels)
                loss2 = criterion(mask, labels)
                loss3 = criterion(m1, labels)
                loss4 = criterion(m2, labels)
                loss5 = criterion(m3, labels)
                loss6 = criterion(m4, labels)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            cd_loss = sum(loss)
            grad_scaler.scale(cd_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 20, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            preds, mask, m1, m2, m3, m4 = net(batch_img1, batch_img2)
            loss1 = criterion(preds, labels)
            loss2 = criterion(mask, labels)
            loss3 = criterion(m1, labels)
            loss4 = criterion(m2, labels)
            loss5 = criterion(m3, labels)
            loss6 = criterion(m4, labels)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            cd_loss = sum(loss)

        epoch_loss += cd_loss
        preds = torch.sigmoid(preds)

        preds = preds.float()
        labels = labels.int().unsqueeze(1)
        metric_collection.forward(preds, labels)

        # clear batch variables from memory
        del batch_img1, batch_img2, labels
    epoch_metrics = metric_collection.compute()  # compute epoch metric
    # 格式化输出指标
    metrics_output = epoch_metrics
    formatted_output = "Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f},IOU: {:.4f}, Accuracy: {:.4f}, ".format(
        metrics_output['precision'].item() * 100,
        metrics_output['recall'].item() * 100,
        metrics_output['f1score'].item() * 100,
        metrics_output['Iou'].item() * 100,
        metrics_output['accuracy'].item() * 100
    )
    print(f"{mode}-epoch {epoch + 1} metrics is:{formatted_output}")
    # 保存指标
    # 将指标保存到文本文件中
    with open("metrics.txt", "a") as file:
        file.write(f"{mode}-epoch {epoch + 1} metrics:\n")
        file.write(formatted_output + "\n")

    epoch_loss /= n_iter
    metric_collection.reset()

    # save best model and adjust learning rate according to learning rate scheduler
    if mode == 'val':
        if epoch_metrics['f1score'] > best_metrics['best_f1score']:
            non_improved_epoch = 0
            best_metrics['best_f1score'] = epoch_metrics['f1score']
            if ph.save_best_model:
                save_model(net, best_f1score_model_path, epoch, 'f1score')
        elif epoch_loss < best_metrics['lowest loss']:
            best_metrics['lowest loss'] = epoch_loss
            if ph.save_best_model:
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
            print("Best model updated!")
            save_model(net, checkpoint_path, epoch, 'checkpoint', optimizer=optimizer)
    print(f"{mode}-epoch {epoch + 1} loss and lr is:{epoch_loss},{lr}")
    if mode == 'train':
        return net, optimizer, grad_scaler, total_step, lr
    elif mode == 'val':
        return net, optimizer, total_step, lr, best_metrics, non_improved_epoch
    else:
        raise NameError('mode should be train or val')
