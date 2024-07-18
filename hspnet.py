
import torch
from torch.cuda.amp import autocast  # type: ignore
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset import build_dataset
from log import logger
from model import load_clip_model, HSPNet
from utils import ModelEma, get_ema_co

from config import cfg  # isort:skip

class HSPNetTrainer():

    def __init__(self) -> None:
        super().__init__()

        clip_model, _ = load_clip_model()
        # image_size = clip_model.visual.input_resolution
        image_size = cfg.image_size

        train_preprocess = transforms.Compose([
            transforms.Resize(image_size,
                              interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        val_preprocess = transforms.Compose([
            transforms.Resize(image_size,
                              interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        train_loader, val_loader = build_dataset(train_preprocess,
                                                 val_preprocess)
        self.train_loader = train_loader
        self.val_loader = val_loader

        classnames = val_loader.dataset.labels()
        assert (len(classnames) == cfg.num_classes)

        self.model = HSPNet(classnames, clip_model)
        self.classnames = classnames
        logger.info("Turning off gradients in the text encoder")
        for name, param in self.model.named_parameters():
            if "text_encoder" in name:
                param.requires_grad_(False)

        self.model.cuda()
        ema_co = get_ema_co()
        logger.info(f"EMA CO: {ema_co}")
        self.ema = ModelEma(self.model, ema_co)  # 0.9997^641=0.82
    
    def train(self, input, target, criterion, epoch, epoch_i) -> torch.Tensor:
        index, image = input
        image = image.cuda()
        with autocast():  # mixed precision
            output = self.model(
                image).float()  # sigmoid will be done in loss !
        if cfg.loss == 'SPLC':
            loss, labels = criterion(output, target, epoch)
        else:
            loss, labels = criterion(output, target)
        return loss