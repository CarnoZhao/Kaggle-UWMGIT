### Introduction

Hello!

Below you can find a outline of how to reproduce my solution for the [UW-Madison GI Tract Image Segmentation | Kaggle](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/337197#1864282)

If you run into any trouble with the setup/code or have any questions please contact me at 273806108@qq.com



### Contents

```sh
preprocess.py: data preprocessing codes
inference.py: inferencing codes
other: necessary codes for `mmsegmentation` and `monai` toolboxes
```



### Hardware

```sh
Ubuntu 16.04 LTS (512 GB boot disk)
48 x Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
126 GB Memory
4 x NVIDIA Titan RTX
```

### Software

```sh
python==3.7.10
CUDA==10.2
cudnn==7.6.5
nvidia-drivers==440.4
(other refer to ./requirements.txt)
```

### Data setup

```sh
# DOWNLOAD DATA
kaggle competitions download -c uw-madison-gi-tract-image-segmentation

mkdir -p ./data/tract
mv uw-madison-gi-tract-image-segmentation.zip ./data/tract

cd ./data/tract
unzip uw-madison-gi-tract-image-segmentation.zip
cd ../..
```


The expected after unzip should be:

```sh
./data/tract
        ├── sample_submission.csv
        ├── test
        ├── train
        ├── train.csv
```
Install base requirements, `mmsegmentation` and `monai` toolboxes

```sh
# INSTALL PYTHON REQUIREMENTS
pip install -r requirements.txt
pip install "monai[ignite,skimage,nibabel]==0.8.1"
pip install mmcv-full==1.3.17 --force-reinstall -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
pip install -v -e .
```

### Data preprocess

```sh
python data_preprocess.py
```

### Training

> NOTE: **make sure internet connection for public pretrained weights downloading**

```sh
mkdir -p saved_weights/cls saved_weights/seg saved_weights/3d

# DOWNLOAD PRETRAINED WEIGHTS
mkdir weights
cd weights
wget https://dl.fbaipublicfiles.com/convnext/ade20k/convnext_base_22k_224.pth
wget https://dl.fbaipublicfiles.com/convnext/ade20k/convnext_small_1k_224_ema.pth
cd ..

# TRAIN CLASSIFICATION MODELS
id=1
for config in $(find ./work_configs/tract/final_solution/classification_configs/cls*.py | sort); do
	./tools/dist_train.sh $config 2
	last_work_dir=$(ls ./work_dirs/tract/ -rt | tail -n 1)
	last_weight=$(ls ./work_dirs/tract/$last_work_dir/*.pth -rt | tail -n 1)
	last_config=$(ls ./work_dirs/tract/$last_work_dir/*.py -rt | tail -n 1)
	mv ./work_dirs/tract/$last_work_dir/$last_weight ./saved_weights/cls/cls_${id}.pth
	mv ./work_dirs/tract/$last_work_dir/$last_config ./saved_weights/cls/cls_${id}.py
	id=$[id+1]
done

# TRAIN SEGMENTATION MODELS
id=1
for config in $(find ./work_configs/tract/final_solution/segmentation_configs/seg*.py | sort); do
	./tools/dist_train.sh $config 2
	last_work_dir=$(ls ./work_dirs/tract/ -rt | tail -n 1)
	last_weight=$(ls ./work_dirs/tract/$last_work_dir/*.pth -rt | tail -n 1)
	last_config=$(ls ./work_dirs/tract/$last_work_dir/*.py -rt | tail -n 1)
	mv ./work_dirs/tract/$last_work_dir/$last_weight ./saved_weights/seg/seg_${id}.pth
	mv ./work_dirs/tract/$last_work_dir/$last_config ./saved_weights/seg/seg_${id}.py
	id=$[id+1]
done

# TRAIN 3D MODELS
cd ./monai
fold=-1
for n in (12 20 32); do
    mkdir -p  ./output/segres${n}_all/all
    python multilabel_train.py \
        -c segres${n}_all \
        -f $fold \
        > ./output/segres${n}_all/all/output.txt
        
    mkdir -p  ./output/segres${n}_all_round2/all
    python multilabel_train.py \
        -c segres${n}_all_round2 \
        -f $fold \
        -w ./output/segres${n}_all/all/last.pth \
        > ./output/segres${n}_all_round2/all/output.txt
    mv ./output/segres${n}_all_round2/all/last.pth ../saved_weights/3d/segres${n}.pth
done
cd ..
```

### Inferencing

```
python inference.py
```





