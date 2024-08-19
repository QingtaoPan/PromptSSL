# -*- coding: utf-8 -*-
import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn

from nets.text_mask.model_novel import my_novel
from torch.utils.data import DataLoader
import logging
from train_my_one_epoch_spine import train_one_epoch, print_summary
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch
from thop import profile

import h5py
from torch.utils.data import Dataset, DataLoader
import augment.transforms_2d as transforms
import cv2
import torch


def logger_config(log_path):  # 'MoNuSeg/LViT/Test_session_time/Test_session_time.log'
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def choose_model(model_type, model_):
    if model_type == 'student':
        return model_
    elif model_type == 'teacher':
        for param in model_.parameters():
            param.detach_()
        return model_


def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):  # 2,
    # Load train and val data
    # train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])  # 串联图像的多个操作
    # val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    # if config.task_name == 'MoNuSeg':
    #     train_text = read_text(config.train_dataset + 'Train_text.xlsx')  # 训练文本数据 key: value
    #     val_text = read_text(config.test_dataset + 'Test_text.xlsx')  # 验证文本数据 key: values
    #     train_dataset = ImageToImage2D_val(config.train_dataset, config.task_name, train_text, train_tf,
    #                                    image_size=config.img_size)  # {'image': image, 'label': mask, 'text': text}, image_filename
    #     val_dataset = ImageToImage2D_val(config.test_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)  # {'image': image, 'label': mask, 'text': text}, image_filename
    # elif config.task_name == 'Covid19':
    #     # text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
    #     train_text = read_text(config.train_dataset + 'Train_text.xlsx')  # 训练文本数据 key: value
    #     val_text = read_text(config.val_dataset + 'Val_text.xlsx')  # 验证文本数据 key: values
    #     train_dataset = ImageToImage2D_val(config.train_dataset, config.task_name, train_text, train_tf,
    #                                    image_size=config.img_size)
    #     val_dataset = ImageToImage2D_val(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
    # elif config.task_name == 'Bone':
    #     # text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
    #     train_text = read_text(config.train_dataset + 'Train_text.xlsx')  # 训练文本数据 key: value
    #     val_text = read_text(config.val_dataset + 'Val_text.xlsx')  # 验证文本数据 key: values
    #     train_dataset = ImageToImage2D_val(config.train_dataset, config.task_name, train_text, train_tf,
    #                                    image_size=config.img_size)
    #     val_dataset = ImageToImage2D_val(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)

    # train_loader = DataLoader(train_dataset,
    #                           batch_size=config.batch_size,  # 2 config.batch_size
    #                           shuffle=True,  # 每一步打乱顺序
    #                           worker_init_fn=worker_init_fn,
    #                           num_workers=8,
    #                           pin_memory=True)
    #
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=config.batch_size_val,
    #                         shuffle=True,
    #                         worker_init_fn=worker_init_fn,
    #                         num_workers=8,
    #                         pin_memory=True)

    ################### ----------spine-data--------#####################
    class H5Dataset(Dataset):
        """Dataset wrapping data and target tensors.

        Each sample will be retrieved by indexing both tensors along the first
        dimension.

        Arguments:
            data_tensor (Tensor): contains sample data.
            target_tensor (Tensor): contains sample targets (labels).
        """

        def __init__(self, data, return_weight=False, transformer_config=None, phase='train'):
            assert data['mr'].shape[0] == data['weight'].shape[0] == data['mask'].shape[0]
            self.raw = data['mr']
            self.weight = data['weight']
            # self.unary = data['unary']  # --------------------------------------------------------------------------------------------------------------
            self.mask = data['mask']
            self.filename = data['filename']
            print('------------------------------------------')
            print(self.raw, self.mask, self.filename)
            print(self.filename)
            print(self.raw)
            self.return_weight = return_weight
            self.transformer_config = transformer_config
            self.phase = phase

            if self.phase == 'train' and self.transformer_config is not None:
                self.transformer = transforms.get_transformer(transformer_config, mean=0, std=1, phase=phase)
                self.raw_transform = self.transformer.raw_transform()
                if self.return_weight:
                    self.weight_transform = self.transformer.weight_transform()
                self.label_transform = self.transformer.label_transform()
                # self.unary_transform = self.transformer.unary_transform()  # --------------------------------------------------------------------------------------------------------------

        def __getitem__(self, index):
            if self.phase == 'train' and self.transformer_config is not None:
                raw = np.squeeze(self.raw[index])
                raw_transformed = self._transform_image(raw, self.raw_transform)
                raw_transformed = np.expand_dims(raw_transformed, axis=0)
                if self.return_weight:
                    weight_transformed = self._transform_image(self.weight[index], self.weight_transform)
                label_transformed = self._transform_image(self.mask[index], self.label_transform)
                # unary_transformed = self._transform_image(self.unary[index], self.unary_transform)  # --------------------------------------------------------------------------------------------------------------
                if self.return_weight:
                    # return raw_transformed.astype(np.float32), weight_transformed, unary_transformed, label_transformed.astype(np.long)  # -------------------------------------------------------------------------
                    return raw_transformed.astype(np.float32), weight_transformed, label_transformed.astype(np.long), \
                    self.filename[index].decode()
                else:
                    # return raw_transformed.astype(np.float32), unary_transformed, label_transformed.astype(np.long)  # ------------------------------------------------------------------------------------------------
                    return raw_transformed.astype(np.float32), label_transformed.astype(np.long), self.filename[
                        index].decode()
            else:
                if self.return_weight:
                    # return self.raw[index].astype(np.float32), self.weight[index], self.unary[index], self.mask[index].astype(np.long)  # -------------------------------------------------------------------------------
                    return self.raw[index].astype(np.float32), self.weight[index], self.mask[index].astype(np.long), \
                    self.filename[index].decode()
                else:
                    # return self.raw[index].astype(np.float32), self.unary[index], self.mask[index].astype(np.long)  # ---------------------------------------------------------------------------------------------------
                    return self.raw[index].astype(np.float32), self.mask[index].astype(np.long), self.filename[
                        index].decode()

        def __len__(self):
            return self.raw.shape[0]

        @staticmethod
        def _transform_image(dataset, transformer):
            return transformer(dataset)

    default_conf = {
        "pyinn": False,

        'transformer': {
            'train': {
                'raw': [
                    {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'cubic'},
                    {'name': 'ElasticDeformation', 'spline_order': 3},
                    {'name': 'RandomContrast'}
                ],
                'label': [
                    {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'nearest'},
                    {'name': 'ElasticDeformation', 'spline_order': 0}
                ],
                'unary': [
                    {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'cubic'},
                    {'name': 'ElasticDeformation', 'spline_order': 3}
                ],
                'weight': [
                    {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'cubic'},
                    {'name': 'ElasticDeformation', 'spline_order': 3}
                ]
            },
            'test': {
                'raw': None,
                'label': None,
                'unary': None,
                'weight': None
            }
        }
    }
    conf = default_conf

    filePath = '/root/data1/spine/public/data/fine/in/h5py/fold1_data.h5'
    f = h5py.File(filePath, 'r')
    train_data = f['train']

    train_set = H5Dataset(data=train_data, transformer_config=conf['transformer'], return_weight=False,
                          phase='train')  # return_weight-->False、
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=config.batch_size, pin_memory=False, shuffle=True)


    lr = config.learning_rate  # 1e-3
    logger.info(model_type)

    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))  # 4
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))  # 4
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))  # 4
        # model_1 = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)  # 3, 1
        # model_2 = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        model_novel = my_novel(config.batch_size)  # 32

    # elif model_type == 'LViT_pretrain':
    #     config_vit = config.get_CTranS_config()
    #     logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
    #     logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
    #     logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
    #     model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    #     pretrained_UNet_model_path = "MoNuSeg/LViT/Test_session_05.23_10h55/models/best_model-LViT.pth.tar"
    #     pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
    #     pretrained_UNet = pretrained_UNet['state_dict']
    #     model2_dict = model.state_dict()
    #     state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
    #     print(state_dict.keys())
    #     model2_dict.update(state_dict)
    #     model.load_state_dict(model2_dict)
    #     logger.info('Load successful!')

    else:
        raise TypeError('Please enter a valid name for the model type')
    # input = torch.randn(1, 1, 256, 512)
    # input_aug = torch.randn(1, 1, 256, 512)
    # text = 'i love you'
    # flops, params = profile(model_novel, inputs=(input, input_aug, text, ))  # 计算模型复杂度
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))
    ######################################################################################################################
    device = torch.device(config.device)
    # model_1 = model_1.to(device)
    # model_2 = model_2.to(device)
    model_novel = model_novel.to(device)
    # model_1 = nn.DataParallel(model_1, device_ids=[0, 1])  # 并行运算
    # model_2 = nn.DataParallel(model_2, device_ids=[0, 1])
    # model_novel = nn.DataParallel(model_novel, device_ids=[0, 1])

    #------------------------- 模型参数更新 -------------------------------------- #
    # model_student = choose_model(model_type='student', model_=model_1)
    # model_teacher = choose_model(model_type='teacher', model_=model_2)

    #------------------------- 模型参数更新 -------------------------------------- #

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_student.parameters()), lr=lr)  # Choose optimize
    optimizer_novel = torch.optim.Adam(filter(lambda p: p.requires_grad, model_novel.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scaler_novel = torch.cuda.amp.GradScaler(enabled=True)

    # if config.cosineLR is True:
    #     lr_scheduler = CosineAnnealingWarmRestarts(optimizer_novel, T_0=10, T_mult=1, eta_min=1e-4)
    # else:
    #     lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    global_step = 0
    min_spine_loss = 10.0

    for epoch in range(config.epochs):  # loop over the dataset multiple times


        global_step = global_step + epoch*len(train_loader)
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        # model_student.train(True)
        # model_teacher.train(True)
        model_novel.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        spine_loss = train_one_epoch(train_loader, model_novel, criterion, optimizer_novel, writer, epoch, None, model_type, logger, scaler, scaler_novel, global_step)  # sup

        # evaluate on validation set------------------------------------------------------------------------------------
        logger.info('Validation')
        # with torch.no_grad():
        #     model_student.eval()
        #     val_loss, val_dice = train_one_epoch(val_loader, model_student, model_teacher, model_novel, criterion, optimizer, optimizer_novel, writer, epoch, lr_scheduler, model_type, logger, scaler, scaler_novel, global_step)
        # =============================================================
        #       Save best model
        # =============================================================
        if spine_loss < min_spine_loss:
            logger.info(
                '\t Saving best model, spine_loss decrease from: {:.4f} to {:.4f}'.format(min_spine_loss, spine_loss))
            max_dice = 3.33
            save_checkpoint({'epoch': epoch,
                             'best_model': True,
                             'model': model_type,
                             'state_dict': model_novel.state_dict(),# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
                             'val_loss': 3.33,
                             'optimizer': optimizer_novel.state_dict()}, config.model_path)
            torch.save(model_novel, config.model_path + '/' + 'best_model.pth')
            min_spine_loss = spine_loss



        if epoch+1 == config.epochs:
            max_dice = 3.33
            best_epoch = epoch + 1
            save_checkpoint({'epoch': epoch,
                             'best_model': False,
                             'model': model_type,
                             'state_dict': model_novel.state_dict(),# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
                             'val_loss': 3.33,
                             'optimizer': optimizer_novel.state_dict()}, config.model_path)
            torch.save(model_novel, config.model_path + '/' + 'last_model.pth')

        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(3.33, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if epoch == config.epochs:
            logger.info('\tstopping!')
            break

    return model_novel


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model_novel = main_loop(model_type=config.model_name, tensorboard=True)
