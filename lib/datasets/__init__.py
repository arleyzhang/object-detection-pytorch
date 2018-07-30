import torch.utils.data as data
from .det_dataset import detection_collate
from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES
from .coco import COCODetection, COCOAnnotationTransform, get_label_map
from .config import *
from lib.utils.augmentations import SSDAugmentation

dataset_map = {'VOC0712': VOCDetection,
               'COCO2014': COCODetection}


def dataset_factory(phase, cfg):
    det_dataset = dataset_map[cfg.DATASET.NAME]
    if phase == 'train':
        dataset = det_dataset(cfg.DATASET.DATASET_DIR, cfg.DATASET.TRAIN_SETS,
                    SSDAugmentation(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS))
        data_loader = data.DataLoader(dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
                                       num_workers=cfg.DATASET.NUM_WORKERS,
                                       shuffle=True, collate_fn=detection_collate,
                                       pin_memory=True, drop_last=True)
    elif phase == 'eval':
        dataset = det_dataset(cfg.DATASET.DATASET_DIR, cfg.DATASET.TEST_SETS,
                                  SSDAugmentation(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS,
                                                  use_base=True))
        data_loader = data.DataLoader(dataset, batch_size=cfg.DATASET.EVAL_BATCH_SIZE,
                                     num_workers=cfg.DATASET.NUM_WORKERS, shuffle=False,
                                     collate_fn=detection_collate, pin_memory=True)
    else:
        raise Exception("unsupported phase type")
    return data_loader
