import sys
import re
import gc
import time

THRES_HIGH = 12
THRES_LOW = 12

include_2d = True

from glob import glob
from tqdm.notebook import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt  
from skimage import measure
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, SegResNet
from monai.data import CacheDataset, DataLoader
from monai.transforms import *

default_3d_tta = [[2],[3]]
class cfg_3d:
    img_size = (224, 224, 80)
    in_channels = 1
    out_channels = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = [
        dict(
            weight = "./saved_weights/3d/segres12.pth",
            model_type = SegResNet, spatial_dims = 3, in_channels = 1, out_channels = 3, 
            act = 'PRELU', norm ='BATCH',
            init_filters = 12,
            tta = default_3d_tta + [[2,3]],
        ),
        dict(
            weight = "./saved_weights/3d/segres20.pth",
            model_type = SegResNet, spatial_dims = 3, in_channels = 1, out_channels = 3, 
            act = 'PRELU', norm ='BATCH',
            init_filters = 20,
            tta = default_3d_tta,
        ),
        dict(
            weight = "./saved_weights/3d/segres32.pth",
            model_type = SegResNet, spatial_dims = 3, in_channels = 1, out_channels = 3, 
            act = 'PRELU', norm ='BATCH',
            init_filters = 32,
            tta = default_3d_tta,
        ),
    ]
    batch_size = 1
    sw_batch_size = 2
    
test_transforms_3d = Compose(
    [
        AddChanneld(keys="image"), # c, d, h, w
        Transposed(keys="image", indices=[0, 3, 2, 1]), # c, w, h, d
        Lambdad(keys="image", func=lambda x: x / x.max()),
        EnsureTyped(keys="image", dtype=torch.float32),
    ]
)

def get_model(cfg, weight):
    weight_path = weight.pop("weight")
    model = weight.pop("model_type")(**weight)
    stt = torch.load(weight_path)
    if "model" in stt: stt = stt["model"]
    if all([k.startswith("module.") for k in stt]): stt = {k[7:]: v for k, v in stt.items()}
    model.load_state_dict(stt)
    model.eval()
    return model
    
models_3d = []
dims_3d_tta = []

for weight in tqdm(cfg_3d.weights):
    dims_3d_tta.append(weight.pop("tta"))
    model = get_model(cfg_3d, weight)
    models_3d.append(model)

if include_2d:
    from mmseg.apis import init_segmentor, inference_segmentor
    from mmcv.utils import config

    cfgs = [
        dict(
            cfg = "./saved_weights/seg/seg_1.py",
            ckpt = "./saved_weights/seg/seg_1.pth",
            weight = 0.2,
            tta = True,
        ),
        dict(
            cfg = "./saved_weights/seg/seg_2.py",
            ckpt = "./saved_weights/seg/seg_2.pth",
            weight = 0.2,
            tta = True,
        ),
        dict(
            cfg = "./saved_weights/seg/seg_3.py",
            ckpt = "./saved_weights/seg/seg_3.pth",
            weight = 0.3,
            tta = True,
        ),
        dict(
            cfg = "./saved_weights/seg/seg_4.py",
            ckpt = "./saved_weights/seg/seg_4.pth",
            weight = 0.2,
            tta = True,
        ),
        dict(
            cfg = "./saved_weights/seg/seg_5.py",
            ckpt = "./saved_weights/seg/seg_5.pth",
            weight = 0.1,
            tta = True,
        ),
        dict(
            cfg = "./saved_weights/seg/seg_6.py",
            ckpt = "./saved_weights/seg/seg_6.pth",
            weight = 0.1,
            tta = False,
        ),
        dict(
            cfg = "./saved_weights/seg/seg_7.py",
            ckpt = "./saved_weights/seg/seg_7.pth",
            weight = 0.1,
            tta = False,
        ),
    ]

    weights = []
    models = []
    for cfg_dic in cfgs:
        cfg = cfg_dic["cfg"]
        ckpt = cfg_dic["ckpt"]
        weights.append(cfg_dic["weight"])
        
        cfg = config.Config.fromfile(cfg)
        cfg.model.backbone.pretrained = None
        cfg.model.test_cfg.logits = True
        # TTA >>>>>
        cfg.data.test.pipeline[1].flip = cfg_dic["tta"]
        # <<<<<<<<<
        cfg.data.test.pipeline[1].transforms.insert(2, dict(type="Normalize", mean=[0,0,0], std=[1,1,1], to_rgb=False))

        model = init_segmentor(cfg, ckpt)
        models.append(model)
    weights = np.array(weights) / sum(weights)

if include_2d:
    from mmseg.apis import init_segmentor, inference_segmentor
    from mmcv.utils import config
    cfgs = [
        "./saved_weights/cls/cls_1.py",
        "./saved_weights/cls/cls_2.py",
        "./saved_weights/cls/cls_3.py",
        "./saved_weights/cls/cls_4.py",
    ]

    ckpts = [
        "./saved_weights/cls/cls_1.pth",
        "./saved_weights/cls/cls_2.pth",
        "./saved_weights/cls/cls_3.pth",
        "./saved_weights/cls/cls_4.pth",
    ]

    cls_models = []
    for cfg, ckpt in zip(cfgs, ckpts):
        cfg = config.Config.fromfile(cfg)
        cfg.model.backbone.pretrained = None
        cfg.model.test_cfg.logits = True
        
        cfg.data.test.pipeline[1].transforms.insert(2, dict(type="Normalize", mean=[0,0,0], std=[1,1,1], to_rgb=False))

        model = init_segmentor(cfg, ckpt)
        cls_models.append(model)

import os
import cv2
import numpy as np
import pandas as pd
import glob
from tqdm.auto import tqdm
from scipy.ndimage import binary_closing, binary_opening, measurements

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

classes = ['large_bowel', 'small_bowel', 'stomach']
data_dir = "./data/tract/"
test_dir = os.path.join(data_dir, "test")
sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
test_images = glob.glob(os.path.join(test_dir, "**", "*.png"), recursive = True)

if len(test_images) == 0:
    test_dir = os.path.join(data_dir, "train")
    sub = pd.read_csv(os.path.join(data_dir, "train.csv"))[["id", "class"]].iloc[:144 * 3]
    sub["predicted"] = ""
    test_images = glob.glob(os.path.join(test_dir, "**", "*.png"), recursive = True)
    
id2img = {_.rsplit("/", 4)[2] + "_" + "_".join(_.rsplit("/", 4)[4].split("_")[:2]): _ for _ in test_images}
sub["file_name"] = sub.id.map(id2img)
sub["days"] = sub.id.apply(lambda x: "_".join(x.split("_")[:2]))
fname2index = {f + c: i for f, c, i in zip(sub.file_name, sub["class"], sub.index)}
sub


def make_3d_prediction(imgs, cnts_cls):
    im_vol = np.transpose(imgs, [2, 0, 1])
    im_vol = test_transforms_3d({'image': im_vol})['image'].unsqueeze(0).to(cfg_3d.device)

    sliding_args = {"overlap": 0.7, "mode": "gaussian"} if include_2d else {}
    pred_all_3d = 0
    tta_cnt = 0
    for model_idx, model in enumerate(models_3d):
        model.to(cfg_3d.device)
        with torch.no_grad():
            pred_all_3d += sliding_window_inference(
                im_vol, 
                cfg_3d.img_size, 
                cfg_3d.sw_batch_size, 
                model,
                **sliding_args,
            ).cpu().sigmoid()
            tta_cnt += 1
        for dims in dims_3d_tta[model_idx]:
            with torch.no_grad():
                pred_all_3d += torch.flip(
                    sliding_window_inference(
                        torch.flip(im_vol, dims=dims), 
                        cfg_3d.img_size, 
                        cfg_3d.sw_batch_size, 
                        model,
                        **sliding_args,
                    ).cpu().sigmoid(),
                    dims=dims) 
                tta_cnt += 1
        model.to('cpu')
    pred_all_3d = torch.permute(pred_all_3d[0], [3, 0, 2, 1]) / tta_cnt # from (c,w,h,d) -> to (d,c,h,w)
    pred_all_3d = pred_all_3d.numpy()
    del im_vol
    gc.collect()
    return pred_all_3d

def make_2d_classification_predictions(imgs, old_size):
    preds = []; cls_cnts = []
    for i in range(imgs.shape[-1]):
        if include_2d:
            img = imgs[...,[max(0, i - 2), i, min(imgs.shape[-1] - 1, i + 2)]]
            res = []
            new_img = img.astype(np.float32) / img.max()
            res = [inference_segmentor(model, new_img)[0][0] for model in cls_models]
            res = sum(res) / len(res)
            
            cls_cnt = (res > 0.6).astype(np.uint8)
            cls_cnt = cv2.resize(cls_cnt, old_size[::-1], interpolation = cv2.INTER_NEAREST).sum((0,1))
            
            res = (res > 0.5).astype(np.uint8)
            res = cv2.resize(res, old_size[::-1], interpolation = cv2.INTER_NEAREST)
            preds.append(res)
            cls_cnts.append(cls_cnt)
    
    if include_2d:
        preds = np.stack(preds, 0)
        cls_cnts = np.stack(cls_cnts, 0)
    else:
        preds = None; cls_cnts = None
    return preds, cls_cnts

def make_2d_predictions_carnoandnam(imgs, old_size, preds_3d, preds_cls, cnts_cls):
    preds = []
    for i in range(imgs.shape[-1]):
        
        res_3d = preds_3d[i]
        res_3d = np.transpose(res_3d, (1,2,0)) # make channel last (h,w,c)
        
        if include_2d:
            cnt = cnts_cls[i]
            pred_cls = preds_cls[i]
            if (cnt > THRES_HIGH).any():
                img = imgs[...,[max(0, i - 2), i, min(imgs.shape[-1] - 1, i + 2)]]
                new_img = img.astype(np.float32) / img.max()

                res = []
                for model, weight in zip(models, weights):
                    res.append(inference_segmentor(model, new_img)[0][0] * weight)
                res = sum(res) / sum(weights)
                
                res = cv2.resize(res, old_size[::-1], interpolation = cv2.INTER_NEAREST)
                res_ens = 0.6 * res + 0.4 * res_3d
                #### 0.5 thres ####
                # res_ens = (res_ens > 0.5).astype(np.uint8)
                #### 0.4/0.5 thres ####
                res1 = (res_ens > [0.4, 0.4, 0.4]).astype(np.uint8)
                res1[...,res_ens.max((0,1)) < [0.5, 0.5, 0.5]] = 0
                res_ens = res1
                for j in range(3):
                    if cnt[j] < THRES_LOW:
                        res_ens[...,j] = 0
                    elif cnt[j] <= THRES_HIGH:
                        res_ens[...,j] = pred_cls[...,j]
            else:
                res_ens = pred_cls
                for j in range(3):
                    if cnt[j] < THRES_LOW:
                        res_ens[...,j] = 0
        else:
            res_ens = (res_3d > 0.5).astype(int)
        preds.append(res_ens)
    
    preds = np.stack(preds, 0)
    
#     for org in range(3):
#         connect = measure.label(preds[...,org])
#         connect_label, connect_cnt = np.unique(connect, return_counts = True)
#         small_cnt_label = connect_label[connect_cnt < 20]
#         preds[np.isin(connect, small_cnt_label),org] = 0
    
    return preds


show_time = False
subs = []
for day, group in tqdm(sub.groupby("days")):
    imgs = []
    file_names = []
    old_sizes = []
    for file_name in sorted(group.file_name.unique()):
        img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
        old_size = img.shape[:2]
        imgs.append(img)
        file_names.append(file_name)
        old_sizes.append(old_size)
        
    imgs = np.stack(imgs, -1)
    
    ###########################
    #                         #
    #    2.5D classification  #
    #                         #
    ###########################
    if show_time: stamp = time.time()
    preds_cls, cnts_cls = make_2d_classification_predictions(imgs, old_size)
    if show_time: print("2d classification: " , time.time() - stamp)
    
    ###########################
    #                         #
    #      3D prediction      #
    #                         #
    ###########################
    if show_time: stamp = time.time()
    pred_3d = make_3d_prediction(imgs, cnts_cls)
    if show_time: print("3d prediction: ", time.time() - stamp)
    
    ###########################
    #                         #
    #      2.5D prediction    #
    #                         #
    ###########################
    if show_time: stamp = time.time()
    preds = make_2d_predictions_carnoandnam(imgs, old_size, pred_3d, preds_cls, cnts_cls)
    if show_time: print("2d prediction: ", time.time() - stamp)
    
    
    ###########################
    #                         #
    #         output          #
    #                         #
    ###########################
    for i, res in enumerate(preds):
        file_name = file_names[i]
        old_size = old_sizes[i]
        for j in range(3):
            rle = rle_encode(res[...,j])
            index = fname2index[file_name + classes[j]]
            sub.loc[index, "predicted"] = rle
            
    del preds_cls, cnts_cls, imgs, pred_3d, preds
    gc.collect()

sub = sub[["id", "class", "predicted"]]
sub.to_csv("submission.csv", index = False)
sub