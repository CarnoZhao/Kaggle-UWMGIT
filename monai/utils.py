import torch
from torch import nn
from torch.optim import lr_scheduler
from monai.utils import set_determinism
from torch import optim
from monai.data import CacheDataset, DataLoader, ThreadDataLoader, PersistentDataset
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction
from monai.losses import DiceLoss
from monai.transforms import LoadImage
from monai.optimizers.lr_scheduler import WarmupCosineSchedule


class DiceBceMultilabelLoss(_Loss):
    def __init__(
        self,
        w_dice = 0.5,
        w_bce = 0.5,
        reduction = LossReduction.MEAN,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.dice_loss = DiceLoss(sigmoid=True, smooth_nr=0.01, smooth_dr=0.01, include_background=True, batch=True, squared_pred=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
    def forward(self, pred, label):
        
        loss = self.dice_loss(pred, label) * self.w_dice + self.bce_loss(pred, label) * self.w_bce
        return loss


def get_train_dataloader(train_dataset, cfg):

    if cfg.gpu_cache:
        train_dataloader = ThreadDataLoader(
            train_dataset,
            shuffle=True,
            batch_size=cfg.batch_size,
            num_workers=0,
            drop_last=True,
        )
        return train_dataloader 

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    return train_dataloader


def get_val_dataloader(val_dataset, cfg):
    if cfg.val_gpu_cache:
        val_dataloader = ThreadDataLoader(
            val_dataset,
            batch_size=cfg.val_batch_size,
            num_workers=0,
        )
        return val_dataloader

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
    )
    return val_dataloader

def get_train_dataset(cfg):
    train_ds = CacheDataset(
        data=cfg.data_json["train"],
        transform=cfg.train_transforms,
        cache_rate=cfg.train_cache_rate,
        num_workers=cfg.num_workers,
        copy_cache=False,
    )
    # train_ds = PersistentDataset(
    #     data=cfg.data_json["train"],
    #     transform=cfg.train_transforms,
    #     cache_dir="cache_data",
    # )
    return train_ds

def get_val_dataset(cfg):
    val_ds = CacheDataset(
        data=cfg.data_json["val"],
        transform=cfg.val_transforms,
        cache_rate=cfg.val_cache_rate,
        num_workers=cfg.num_workers,
        copy_cache=False,
    )
    return val_ds

def get_val_org_dataset(cfg):
    val_ds = CacheDataset(
        data=cfg.data_json["val"],
        transform=cfg.org_val_transforms,
        cache_rate=cfg.val_cache_rate,
        num_workers=cfg.num_workers,
        copy_cache=False,
    )
    return val_ds

def get_optimizer(model, cfg):

    params = model.parameters()
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    return optimizer

def get_scheduler(cfg, optimizer, total_steps):

    if cfg.lr_mode == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs * (total_steps // cfg.batch_size),
            eta_min=cfg.min_lr,
        )

    elif cfg.lr_mode == "warmup_restart":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.restart_epoch * (total_steps // cfg.batch_size),
            T_mult=1,
            eta_min=cfg.min_lr,
        )

    return scheduler


def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint
