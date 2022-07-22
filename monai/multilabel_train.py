import argparse
import gc
import importlib
import os
import sys
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils import *

from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    Activationsd,
    AsDiscreted,
    KeepLargestConnectedComponentd,
    Invertd,
    LoadImage,
    Transposed,
)
import json
from metric import HausdorffScore
from monai.utils import set_determinism
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet, SegResNet, DynUnet
from monai.optimizers import Novograd
from monai.metrics import DiceMetric
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(cfg):

    # data sequence
    if cfg.fold != -1:
        cfg.data_json_dir = cfg.data_dir + f"dataset_3d_fold_{cfg.fold}.json"
    else:
        cfg.data_json_dir = cfg.data_dir + f"dataset_3d_all.json"

    with open(cfg.data_json_dir, "r") as f:
        cfg.data_json = json.load(f)

    if cfg.fold != -1:
        fold_dir = f"fold{cfg.fold}"
    else:
        fold_dir = "all"
    os.makedirs(str(cfg.output_dir + f"/{fold_dir}/"), exist_ok=True)

    # # set random seed
    # set_determinism(cfg.seed)

    train_dataset = get_train_dataset(cfg)
    train_dataloader = get_train_dataloader(train_dataset, cfg)

    val_dataset = get_val_dataset(cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)

    print(f"run fold {cfg.fold}, train len: {len(train_dataset)}")

    if cfg.model_type.startswith("segres"):
        model = SegResNet(
            spatial_dims = 3,
            in_channels = 1,
            out_channels = 3,
            init_filters = int(cfg.model_type.replace("segres", "")),
            norm = "BATCH",
            act = "PRELU"
        ).to(cfg.device)


    print(cfg.weights)
    if cfg.weights is not None:
        stt = torch.load(cfg.weights, map_location = "cpu")
        if "model" in stt:
            stt = stt["model"]
        if "state_dict" in stt:
            stt = stt["state_dict"]
            del stt["out.conv.conv.weight"], stt["out.conv.conv.bias"]
        model.load_state_dict(stt, strict = False)
        print(f"weights from: {cfg.weights} are loaded.")

    # set optimizer, lr scheduler
    total_steps = len(train_dataset)
    optimizer = get_optimizer(model, cfg)
    # optimizer = Novograd(model.parameters(), cfg.lr)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    seg_loss_func = DiceBceMultilabelLoss(w_dice=cfg.w_dice, w_bce=1-cfg.w_dice)
    # seg_loss_func = DiceLoss(sigmoid=True, smooth_nr=0.01, smooth_dr=0.01, include_background=True, batch=True)
    dice_metric = DiceMetric(reduction="mean")
    hausdorff_metric = HausdorffScore(reduction="mean")
    metric_function = [dice_metric, hausdorff_metric]

    post_pred = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5),
    ])

    # train and val loop
    step = 0
    i = 0
    if cfg.eval is True:
        best_val_metric = run_eval(
            model=model,
            val_dataloader=val_dataloader,
            post_pred=post_pred,
            metric_function=metric_function,
            seg_loss_func=seg_loss_func,
            cfg=cfg,
            epoch=0,
        )
    else:
        best_val_metric = 0.0
    best_weights_name = "best_weights"
    for epoch in range(cfg.epochs):
        print("EPOCH:", epoch)
        gc.collect()
        if cfg.train is True:
            run_train(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                seg_loss_func=seg_loss_func,
                cfg=cfg,
                # writer=writer,
                epoch=epoch,
                step=step,
                iteration=i,
            )

        if (epoch + 1) % cfg.eval_epochs == 0 and cfg.eval is True and epoch > cfg.start_eval_epoch:
            val_metric = run_eval(
                model=model,
                val_dataloader=val_dataloader,
                post_pred=post_pred,
                metric_function=metric_function,
                seg_loss_func=seg_loss_func,
                cfg=cfg,
                epoch=epoch,
            )

            if val_metric > best_val_metric:
                print(f"Find better metric: val_metric {best_val_metric:.5} -> {val_metric:.5}")
                best_val_metric = val_metric
                checkpoint = create_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    scheduler=scheduler,
                )
                torch.save(
                    checkpoint,
                    f"{cfg.output_dir}/{fold_dir}/{best_weights_name}.pth",
                )
            else:
                if cfg.load_best_weights is True:
                    try:
                        model.load_state_dict(torch.load(f"{cfg.output_dir}/{fold_dir}/{best_weights_name}.pth")["model"])
                        print(f"metric no improve, load the saved best weights with score: {best_val_metric}.")
                    except:
                        pass

        if (epoch + 1) == cfg.epochs:
            # save final best weights, with its distinct name in order to avoid mistakes.
            if os.path.exists(f"{cfg.output_dir}/{fold_dir}/{best_weights_name}.pth"):
                shutil.copyfile(
                    f"{cfg.output_dir}/{fold_dir}/{best_weights_name}.pth",
                    f"{cfg.output_dir}/{fold_dir}/{best_weights_name}_{best_val_metric:.4f}.pth",
                )

        torch.save(
            model.state_dict(),
            f"{cfg.output_dir}/{fold_dir}/last.pth",
        )
            

def run_train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    seg_loss_func,
    cfg,
    # writer,
    epoch,
    step,
    iteration,
):
    model.train()
    scaler = GradScaler()
    progress_bar = tqdm(range(len(train_dataloader)))
    tr_it = iter(train_dataloader)
    dataset_size = 0
    running_loss = 0.0

    for itr in progress_bar:
        iteration += 1
        batch = next(tr_it)
        inputs, masks = (
            batch["image"].to(cfg.device),
            batch["mask"].to(cfg.device),
        )

        step += cfg.batch_size

        if cfg.amp is True:
            with autocast():
                outputs = model(inputs)
                loss = seg_loss_func(outputs, masks)
        else:
            outputs = model(inputs)
            loss = seg_loss_func(outputs, masks)
        if cfg.amp is True:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        scheduler.step()
        
        running_loss += (loss.item() * cfg.batch_size)
        dataset_size += cfg.batch_size
        losses = running_loss / dataset_size
        progress_bar.set_description(f"loss: {losses:.4f} lr: {optimizer.param_groups[0]['lr']:.6f}")
        del batch, inputs, masks, outputs, loss
    print(f"Train loss: {losses:.4f}")
    torch.cuda.empty_cache()

def run_eval(model, val_dataloader, post_pred, metric_function, seg_loss_func, cfg, epoch):

    model.eval()

    dice_metric, hausdorff_metric = metric_function

    progress_bar = tqdm(range(len(val_dataloader)))
    val_it = iter(val_dataloader)
    with torch.no_grad():
        for itr in progress_bar:
            batch = next(val_it)
            val_inputs, val_masks = (
                batch["image"].to(cfg.device),
                batch["mask"].to(cfg.device),
            )
            if cfg.val_amp is True:
                with autocast():
                    val_outputs = sliding_window_inference(val_inputs, cfg.roi_size, cfg.sw_batch_size, model)
            else:
                val_outputs = sliding_window_inference(val_inputs, cfg.roi_size, cfg.sw_batch_size, model)
            # cal metric
            if cfg.run_tta_val is True:
                tta_ct = 1
                for dims in [[2],[3],[2,3]]:
                    flip_val_outputs = sliding_window_inference(torch.flip(val_inputs, dims=dims), cfg.roi_size, cfg.sw_batch_size, model)
                    val_outputs += torch.flip(flip_val_outputs, dims=dims)
                    tta_ct += 1
                
                val_outputs /= tta_ct

            val_outputs = [post_pred(i) for i in val_outputs]
            val_outputs = torch.stack(val_outputs)
            # metric is slice level put (n, c, h, w, d) to (n, d, c, h, w) to (n*d, c, h, w)
            val_outputs = val_outputs.permute([0, 4, 1, 2, 3]).flatten(0, 1)
            val_masks = val_masks.permute([0, 4, 1, 2, 3]).flatten(0, 1)

            hausdorff_metric(y_pred=val_outputs, y=val_masks)
            dice_metric(y_pred=val_outputs, y=val_masks)

            del val_outputs, val_inputs, val_masks, batch

    dice_score = dice_metric.aggregate().item()
    hausdorff_score = hausdorff_metric.aggregate().item()
    dice_metric.reset()
    hausdorff_metric.reset()

    all_score = dice_score * 0.4 + hausdorff_score * 0.6
    print(f"dice_score: {dice_score} hausdorff_score: {hausdorff_score} all_score: {all_score}")
    torch.cuda.empty_cache()

    return all_score


if __name__ == "__main__":

    sys.path.append("configs")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-c", "--config", default="cfg_unet_multilabel", help="config filename")
    parser.add_argument("-f", "--fold", type=int, default=0, help="fold")
    parser.add_argument("-s", "--seed", type=int, default=20220421, help="seed")
    parser.add_argument("-w", "--weights", default=None, help="the path of weights")

    parser_args, _ = parser.parse_known_args(sys.argv)

    cfg = importlib.import_module(parser_args.config).cfg
    cfg.fold = parser_args.fold
    cfg.seed = parser_args.seed
    cfg.weights = parser_args.weights

    main(cfg)
