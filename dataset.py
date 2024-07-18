import os

import torch
from torchvision.transforms import Compose

from config import cfg
from log import logger
from utils import COCO_missing_dataset, COCO_missing_val_dataset, CocoDetection


def build_dataset(train_preprocess: Compose,
                  val_preprocess: Compose,
                  pin_memory=True):
    if "coco" in cfg.data:
        logger.info("Building coco dataset...")
        return build_coco_dataset(train_preprocess, val_preprocess, pin_memory)
    elif "nuswide" in cfg.data:
        logger.info("Buildding nuswide dataset...")
        return build_nuswide_dataset(train_preprocess, val_preprocess,
                                     pin_memory)
    elif "voc" in cfg.data:
        logger.info("Buildding voc dataset...")
        return build_voc_dataset(train_preprocess, val_preprocess, pin_memory)
    elif "cub" in cfg.data:
        logger.info("Buildding cub dataset...")
        return build_cub_dataset(train_preprocess, val_preprocess, pin_memory)
    else:
        assert (False)


def build_coco_dataset(train_preprocess: Compose,
                       val_preprocess: Compose,
                       pin_memory=True):
    # COCO Data loading
    instances_path_val = os.path.join(cfg.data,
                                      'annotations/instances_val2014.json')
    # instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    instances_path_train = cfg.dataset

    data_path_val = f'{cfg.data}/val2014'  # args.data
    data_path_train = f'{cfg.data}/train2014'  # args.data
    val_dataset = CocoDetection(data_path_val, instances_path_val,
                                val_preprocess)
    train_dataset = COCO_missing_dataset(data_path_train,
                                         instances_path_train,
                                         train_preprocess,
                                         class_num=cfg.num_classes)

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False)

    logger.info("Build dataset done.")
    return [train_loader, val_loader]


def build_voc_dataset(train_preprocess: Compose,
                      val_preprocess: Compose,
                      pin_memory=True):
    # VOC Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}VOC2012/JPEGImages'  # args.data
    data_path_train = f'{cfg.data}VOC2012/JPEGImages'  # args.data

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = COCO_missing_dataset(data_path_train,
                                         instances_path_train,
                                         train_preprocess,
                                         class_num=cfg.num_classes)
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.workers,
                                               pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False)
    logger.info("Build dataset done.")
    return [train_loader, val_loader]


def build_nuswide_dataset(train_preprocess: Compose,
                          val_preprocess: Compose,
                          pin_memory=True):
    # Nus_wide Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}images'  # args.data
    data_path_train = f'{cfg.data}images'  # args.data

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = COCO_missing_dataset(data_path_train,
                                         instances_path_train,
                                         train_preprocess,
                                         class_num=cfg.num_classes)
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.workers,
                                               pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False)
    logger.info("Build dataset done.")
    return [train_loader, val_loader]

def build_cub_dataset(train_preprocess: Compose,
                          val_preprocess: Compose,
                          pin_memory=True):
    # Nus_wide Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}CUB_200_2011/images'  # args.data
    data_path_train = f'{cfg.data}CUB_200_2011/images'  # args.data

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = COCO_missing_dataset(data_path_train,
                                         instances_path_train,
                                         train_preprocess,
                                         class_num=cfg.num_classes)
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.workers,
                                               pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False)
    logger.info("Build dataset done.")
    return [train_loader, val_loader]

