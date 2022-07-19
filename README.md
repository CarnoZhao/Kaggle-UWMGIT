# GUIDE

1. download data from kaggle and unzip all to `./data/tract`

2. run `./data/tract/data.ipynb` under `./data/tract` to convert directory to `mmsegmentation` format (run `mkdir` if necessary)

3. run `./tools/dist_train.sh ./work_configs/tract/baseline.py $NGPUS`, `$NGPUS` depends on your own devices

4. make a kaggle submission notebook refering `./data/tract/submission.ipynb`


# UPDATE

[Competition solution](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/337197) and [submission code](https://www.kaggle.com/code/carnozhao/1st-solution-for-uw?scriptVersionId=100815725) have been open sourced at Kaggle  

aAll config files have been uploaded to `./data/tract/final_solution`, saved in `classification_configs` and `segmentation_configs` respectively. These configs are for **referencing only**. To reproduce, there is much to do with data, including data cleaning and preprocessing, so I have no plan to include this part.

3D monai codes are saved at `./monai`, with only slight changes compared with [public monai codes](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/325646). 
