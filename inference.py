import sys

from torch.utils.data import DataLoader

from Ablation.BaseLine import BaseLine
from Ablation.BaseLine_M1 import BaseLine_M1
from Ablation.BaseLine_M2 import BaseLine_M2
from MCPRNet.MCPRNet import MCPRNet
from MyNet.MDGNetV5 import MDGNetv5
from MyNet.PFNetv2 import PFNetV2
from MyNet.PFNetv3 import PFNetV3
from MyNet.PFNetv4 import PFNetV4
from MyNet.PFNetv5 import PFNetV5
from MyNet.PFNetv6 import PFNetV6
from MyNet.PFNetv6_test import PFNetV6_test

from utils.data_loading import BasicDataset
import logging
from utils.path_hyperparameter import ph
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, JaccardIndex
from utils.dataset_process import compute_mean_std
from tqdm import tqdm


def Test_net(dataset_name, load_checkpoint=False):
    # 1. Create dataset
    dataset_list = ['A', 'B', 'OUT']
    # compute mean and std of train dataset to normalize train/val/test dataset
    # t1_mean, t1_std = compute_mean_std(images_dir=f'{dataset_name}/train/{dataset_list[0]}/')
    # t2_mean, t2_std = compute_mean_std(images_dir=f'{dataset_name}/train/{dataset_list[1]}/')
    # levir
    # t1_mean, t1_std = [0.45026044, 0.44666811, 0.38134658], [0.17456748, 0.16490024, 0.15318057]
    # t2_mean, t2_std = [0.34552285, 0.33819558, 0.28881546], [0.12937804, 0.12601846, 0.1187869]
    # whu
    t1_mean, t1_std = [0.48435662, 0.44378854, 0.38627573], [0.14271143, 0.13803999, 0.13800791]
    t2_mean, t2_std = [0.48289366, 0.48230296, 0.45964595], [0.17929289, 0.16960935, 0.17576571]

    # sysu
    # t1_mean, t1_std = [0.39659575, 0.52846196, 0.46540029], [0.20213537, 0.15811189, 0.15296703]
    # t2_mean, t2_std = [0.40202364, 0.48766127, 0.39895688], [0.18235275, 0.15682769, 0.1543715]

    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)
    test_dataset = BasicDataset(t1_images_dir=f'{dataset_name}/test/{dataset_list[0]}/',
                                t2_images_dir=f'{dataset_name}/test/{dataset_list[1]}/',
                                labels_dir=f'{dataset_name}/test/{dataset_list[2]}/',
                                train=False, **dataset_args)
    # 2. Create data loaders
    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True
                       )
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 3. Initialize logging
    logging.basicConfig(level=logging.INFO)

    # 4. Set up device, model, metric calculator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Using device {device}')
    # 测试
    # net = BaseLine()
    net = PFNetV6()
    # net = MCPRNet()
    # net = PFNetV3()
    # net = PFNetV4()
    net.to(device=device)

    assert ph.load, 'Loading model error, checkpoint ph.load'
    load_model = torch.load(ph.load, map_location=device)
    if load_checkpoint:
        net.load_state_dict(load_model['net'])
    else:
        net.load_state_dict(load_model)
    # ph.load 改成路径
    logging.info(f'Model loaded from {ph.load}')
    torch.save(net.state_dict(), f'{dataset_name}_best_model.pth')

    metric_collection = MetricCollection({
        'accuracy': Accuracy(task="binary").to(device=device),
        'precision': Precision(task="binary").to(device=device),
        'recall': Recall(task="binary").to(device=device),
        'f1score': F1Score(task="binary").to(device=device),
        'Iou': JaccardIndex(task="binary").to(device=device)
    })  # metrics calculator

    net.eval()
    logging.info('SET model mode to test!')
    with torch.no_grad():
        for batch_img1, batch_img2, labels, name in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            labels = labels.float().to(device)

            cd_preds = net(batch_img1, batch_img2, log=True, img_name=name)
            # 变成张量
            # 多输出
            cd_preds = torch.sigmoid(cd_preds[0])
            # 单输出单通道
            # cd_preds = torch.sigmoid(cd_preds)
            # 单输出双通道
            # cd_preds = torch.softmax(cd_preds, dim=1)
            # cd_preds = torch.argmax(cd_preds, dim=1, keepdim=True)  # [32, 1, 256, 256]
            # Calculate and log other batch metrics
            cd_preds = cd_preds.float()
            labels = labels.int().unsqueeze(1)
            metric_collection.update(cd_preds, labels)
        test_metrics1 = metric_collection.compute()
        # 格式化输出指标
        formatted_output = "Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}, IOU: {:.4f}, Accuracy: {:.4f}".format(
            test_metrics1['precision'].item(), test_metrics1['recall'].item(), test_metrics1['f1score'].item(),
            test_metrics1['Iou'].item(),
            test_metrics1['accuracy'].item())

        print(f"Test Metrics is : {formatted_output}")
        metric_collection.reset()

    print('Test over !')


if __name__ == '__main__':
    try:
        Test_net(dataset_name=fr'{ph.dataset_name}', load_checkpoint=False)
    except KeyboardInterrupt:
        logging.info('Error')
        sys.exit(0)
