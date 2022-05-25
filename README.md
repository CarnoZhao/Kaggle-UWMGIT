# GUIDE

1. download data from kaggle and unzip all to `./data/tract`

2. run `./data/tract/data.ipynb` under `./data/tract` to convert directory to `mmsegmentation` format (run `mkdir` if necessary)

3. run `./tools/dist_train.sh ./work_configs/tract/baseline.py $NGPUS`, `$NGPUS` depends on your own devices

4. make a kaggle submission notebook refering `./data/tract/submission.ipynb`


# TODO

full code will be updated after competition ends
