# U-Net: Convolutional Networks for Biomedical Image Segmentation

## Introduction

<!-- [ALGORITHM] -->

<a href="http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/unet.py#L225">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at [this http URL](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142902977-20fe689d-a147-4d92-9690-dbfde8b68dbe.png" width="70%"/>
</div>

<details>
<summary align="right"><a href="https://arxiv.org/abs/1505.04597">UNet (MICCAI'2016/Nat. Methods'2019)</a></summary>

```latex
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

</details>

## Results and models

### DRIVE

|  Method   | Backbone      | Image Size | Crop Size | Stride | Lr schd | Mem (GB) | Inf time (fps) |  Dice | config                                                                                                                   | download                                                                                                                                                                                                                                                                                                                                             |
| ----------- | --------- | ---------- | --------- | -----: | ------- | -------- | -------------: | ----: | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  FCN |  UNet-S5-D16      | 584x565    | 64x64     |  42x42 | 40000   | 0.680    |              - | 78.67 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/fcn_unet_s5-d16_64x64_40k_drive.py)       | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_64x64_40k_drive/fcn_unet_s5-d16_64x64_40k_drive_20201223_191051-5daf6d3b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_64x64_40k_drive/unet_s5-d16_64x64_40k_drive-20201223_191051.log.json)                         |
|  PSPNet |  UNet-S5-D16   | 584x565    | 64x64     |  42x42 | 40000   | 0.599    |              - | 78.62 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/pspnet_unet_s5-d16_64x64_40k_drive.py)    | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_64x64_40k_drive/pspnet_unet_s5-d16_64x64_40k_drive_20201227_181818-aac73387.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_64x64_40k_drive/pspnet_unet_s5-d16_64x64_40k_drive-20201227_181818.log.json)             |
|  DeepLabV3 | UNet-S5-D16  | 584x565    | 64x64     |  42x42 | 40000   | 0.596    |              - | 78.69 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/deeplabv3_unet_s5-d16_64x64_40k_drive.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_64x64_40k_drive/deeplabv3_unet_s5-d16_64x64_40k_drive_20201226_094047-0671ff20.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_64x64_40k_drive/deeplabv3_unet_s5-d16_64x64_40k_drive-20201226_094047.log.json) |

### STARE

|   Method   | Backbone      | Image Size | Crop Size | Stride | Lr schd | Mem (GB) | Inf time (fps) |  Dice | config                                                                                                                     | download                                                                                                                                                                                                                                                                                                                                                     |
| ----------- | --------- | ---------- | --------- | -----: | ------- | -------- | -------------: | ----: | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|  FCN |   UNet-S5-D16     | 605x700    | 128x128   |  85x85 | 40000   | 0.968    |              - | 81.02 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/fcn_unet_s5-d16_128x128_40k_stare.py)       | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_128x128_40k_stare/fcn_unet_s5-d16_128x128_40k_stare_20201223_191051-7d77e78b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_128x128_40k_stare/unet_s5-d16_128x128_40k_stare-20201223_191051.log.json)                         |
|  PSPNet |  UNet-S5-D16   | 605x700    | 128x128   |  85x85 | 40000   | 0.982    |              - | 81.22 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/pspnet_unet_s5-d16_128x128_40k_stare.py)    | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_128x128_40k_stare/pspnet_unet_s5-d16_128x128_40k_stare_20201227_181818-3c2923c4.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_128x128_40k_stare/pspnet_unet_s5-d16_128x128_40k_stare-20201227_181818.log.json)             |
|  DeepLabV3 | UNet-S5-D16  | 605x700    | 128x128   |  85x85 | 40000   | 0.999    |              - | 80.93 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/deeplabv3_unet_s5-d16_128x128_40k_stare.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_128x128_40k_stare/deeplabv3_unet_s5-d16_128x128_40k_stare_20201226_094047-93dcb93c.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_128x128_40k_stare/deeplabv3_unet_s5-d16_128x128_40k_stare-20201226_094047.log.json) |

### CHASE_DB1

| Method    | Backbone      | Image Size | Crop Size | Stride | Lr schd | Mem (GB) | Inf time (fps) |  Dice | config                                                                                                                     | download                                                                                                                                                                                                                                                                                                                                                     |
| ----------- | --------- | ---------- | --------- | -----: | ------- | -------- | -------------: | ----: | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|  FCN |   UNet-S5-D16     | 960x999    | 128x128   |  85x85 | 40000   | 0.968    |              - | 80.24 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/fcn_unet_s5-d16_128x128_40k_chase_db1.py)       | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_128x128_40k_chase_db1/fcn_unet_s5-d16_128x128_40k_chase_db1_20201223_191051-11543527.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_128x128_40k_chase_db1/unet_s5-d16_128x128_40k_chase_db1-20201223_191051.log.json)                         |
|  PSPNet | UNet-S5-D16    | 960x999    | 128x128   |  85x85 | 40000   | 0.982    |              - | 80.36 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/pspnet_unet_s5-d16_128x128_40k_chase_db1.py)    | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_128x128_40k_chase_db1/pspnet_unet_s5-d16_128x128_40k_chase_db1_20201227_181818-68d4e609.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_128x128_40k_chase_db1/pspnet_unet_s5-d16_128x128_40k_chase_db1-20201227_181818.log.json)             |
| DeepLabV3 | UNet-S5-D16 | 960x999    | 128x128   |  85x85 | 40000   | 0.999    |              - | 80.47 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/deeplabv3_unet_s5-d16_128x128_40k_chase_db1.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_128x128_40k_chase_db1/deeplabv3_unet_s5-d16_128x128_40k_chase_db1_20201226_094047-4c5aefa3.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_128x128_40k_chase_db1/deeplabv3_unet_s5-d16_128x128_40k_chase_db1-20201226_094047.log.json) |

### HRF

|  Method    | Backbone      | Image Size | Crop Size | Stride | Lr schd | Mem (GB) | Inf time (fps) |  Dice | config                                                                                                                     | download                                                                                                                                                                                                                                                                                                                                                     |
| ----------- | --------- | ---------- | --------- | -----: | ------- | -------- | -------------: | ----: | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|  FCN |    UNet-S5-D16    | 2336x3504  | 256x256   | 170x170 | 40000   | 2.525    |              - | 79.45 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/fcn_unet_s5-d16_256x256_40k_hrf.py)       | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_256x256_40k_hrf/fcn_unet_s5-d16_256x256_40k_hrf_20201223_173724-d89cf1ed.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/unet_s5-d16_256x256_40k_hrf/unet_s5-d16_256x256_40k_hrf-20201223_173724.log.json)                         |
|  PSPNet |  UNet-S5-D16   | 2336x3504  | 256x256   | 170x170 | 40000   | 2.588    |              - | 80.07 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/pspnet_unet_s5-d16_256x256_40k_hrf.py)    | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_256x256_40k_hrf/pspnet_unet_s5-d16_256x256_40k_hrf_20201227_181818-fdb7e29b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/pspnet_unet_s5-d16_256x256_40k_hrf/pspnet_unet_s5-d16_256x256_40k_hrf-20201227_181818.log.json)             |
|  DeepLabV3 | UNet-S5-D16 | 2336x3504  | 256x256   | 170x170 | 40000   | 2.604    |              - | 80.21 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/deeplabv3_unet_s5-d16_256x256_40k_hrf.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_256x256_40k_hrf/deeplabv3_unet_s5-d16_256x256_40k_hrf_20201226_094047-3a1fdf85.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_256x256_40k_hrf/deeplabv3_unet_s5-d16_256x256_40k_hrf-20201226_094047.log.json) |
