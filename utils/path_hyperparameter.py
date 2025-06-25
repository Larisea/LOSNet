class Path_Hyperparameter:
    random_seed = 42

    # dataset hyper-parameter
    # dataset_name = r'E:/lsj/DATA/CDD/'
    # dataset_name = r'E:/lsj/DATA/LEVIRCD/'
    dataset_name = r'E:/lsj/DATA/WHUCD/'
    # dataset_name = r'E:/lsj/DATA/SYSU/'
    # dataset_name = r'E:/lsj/DATA/samples/'

    # training hyper-parameter
    epochs: int = 100  # Number of epochs
    batch_size: int = 16  # Batch size
    # val batchsize VS train 翻一倍
    inference_ratio = 2  # batch_size in val and test equal to batch_size*inference_ratio
    learning_rate: float = 2e-4  # Learning rate
    factor = 0.1  # learning rate decreasing factor
    patience = 12  # schedular patience
    warm_up_step = 500  # warm up step
    weight_decay: float = 1e-3  # AdamW optimizer weight decay
    amp: bool = True  # if use mixed precision or not
    '''
        测试的时候改成模型路径，训练的时候改成false                  
    '''
    # Load model and/or optimizer from a .pth file for testing or continuing training
    load: str = False
    # load: str = r"E:\lsj\DATA\LEVIRCD\_best_f1score_model\best_f1score_epoch70_Fri_Nov__8_11_16_55_2024.pth"
    # load: str = r"E:\lsj\DATA\LEVIRCD\_checkpoint\checkpoint_epoch79_Thu_Nov__7_19_36_44_2024.pth"
    # load: str = r"E:\lsj\DATA\WHUCD\_checkpoint\checkpoint_epoch39_Mon_Nov_25_16_58_56_2024.pth"
    # load: str = r"E:\lsj\DATA\WHUCD\_best_f1score_model\best_f1score_epoch80_Thu_Nov_28_10_59_17_2024.pth"
    # load: str = r"E:\lsj\DATA\SYSU\_best_f1score_model\best_f1score_epoch10_Mon_Nov_25_09_23_08_2024.pth"

    max_norm: float = 20  # gradient clip max norm

    # evaluate hyper-parameter
    evaluate_epoch: int = 5  # start evaluate after training for evaluate epochs
    stage_epoch = [0, 0, 0, 0, 0]  # adjust learning rate after every stage epoch
    save_checkpoint: bool = True  # if save checkpoint of model or not
    save_interval: int = 20  # save checkpoint every interval epoch
    save_best_model: bool = True  # if save best model or not

    # log wandb hyper-parameter
    log_wandb_project: str = 'dpcd_2'  # wandb project name

    # data transform hyper-parameter
    noise_p: float = 0.8  # probability of adding noise

    # model hyper-parameter
    dropout_p: float = 0.3  # probability of dropout
    patch_size: int = 256  # size of input image

    y = 2  # ECA-net parameter
    b = 1  # ECA-net parameter

    # inference parameter
    log_path = './log_feature/'

    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Path_Hyperparameter.__dict__.items() \
                if not k.startswith('_')}


ph = Path_Hyperparameter()
