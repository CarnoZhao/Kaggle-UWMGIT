import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import glob
from tqdm.auto import tqdm
import albumentations as A

def rle_decode(mask_rle, shape):
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths
    h, w = shape
    img = np.zeros((h * w,), dtype = np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape)

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

os.chdir("./data/tract")

df_train = pd.read_csv("./train.csv")
df_train = df_train.sort_values(["id", "class"]).reset_index(drop = True)
df_train["patient"] = df_train.id.apply(lambda x: x.split("_")[0])
df_train["days"] = df_train.id.apply(lambda x: "_".join(x.split("_")[:2]))
num_slices = len(np.unique(df_train.id))
num_empty_slices = df_train.groupby("id").apply(lambda x: x.segmentation.isna().all()).sum()
num_patients = len(np.unique(df_train.patient))
num_days = len(np.unique(df_train.days))
print({
    "#slices:": num_slices,
    "#empty slices:": num_empty_slices,
    "#patients": num_patients,
    "#days": num_days
})

all_image_files = sorted(glob.glob("./train/*/*/scans/*.png"), key = lambda x: x.split("/")[3] + "_" + x.split("/")[5])
size_x = [int(os.path.basename(_)[:-4].split("_")[-4]) for _ in all_image_files]
size_y = [int(os.path.basename(_)[:-4].split("_")[-3]) for _ in all_image_files]
spacing_x = [float(os.path.basename(_)[:-4].split("_")[-2]) for _ in all_image_files]
spacing_y = [float(os.path.basename(_)[:-4].split("_")[-1]) for _ in all_image_files]
df_train["image_files"] = np.repeat(all_image_files, 3)
df_train["spacing_x"] = np.repeat(spacing_x, 3)
df_train["spacing_y"] = np.repeat(spacing_y, 3)
df_train["size_x"] = np.repeat(size_x, 3)
df_train["size_y"] = np.repeat(size_y, 3)
df_train["slice"] = np.repeat([int(os.path.basename(_)[:-4].split("_")[-5]) for _ in all_image_files], 3)
df_train

os.system("mkdir -p ./mmseg_train/images ./mmseg_train/labels")
for day, group in tqdm(df_train.groupby("days")):
    patient = group.patient.iloc[0]
    imgs = []
    msks = []
    for file_name in tqdm(group.image_files.unique(), leave = False):
        img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
        segms = group.loc[group.image_files == file_name]
        masks = {}
        for segm, label in zip(segms.segmentation, segms["class"]):
            if not pd.isna(segm):
                mask = rle_decode(segm, img.shape[:2])
                masks[label] = mask
            else:
                masks[label] = np.zeros(img.shape[:2], dtype = np.uint8)
        masks = np.stack([masks[k] for k in sorted(masks)], -1)
        imgs.append(img)
        msks.append(masks)
        
    imgs = np.stack(imgs, 0)
    msks = np.stack(msks, 0)
    for i in range(msks.shape[0]):
        img = imgs[[
            max(0, i - 2), 
            i, 
            min(imgs.shape[0] - 1, i + 2)
        ]].transpose(1, 2, 0)
        msk = msks[i]
        new_image_name = f"{day}_{i}.png"
        cv2.imwrite(f"./mmseg_train/images/{new_image_name}", img)
        cv2.imwrite(f"./mmseg_train/labels/{new_image_name}", msk)

os.system("mkdir -p ./mmseg_train/splits")
all_image_files = glob.glob("./mmseg_train/images/*")
patients = [os.path.basename(_).split("_")[0] for _ in all_image_files]


from sklearn.model_selection import GroupKFold

split = list(GroupKFold(5).split(patients, groups = patients))

for fold, (train_idx, valid_idx) in enumerate(split):
    with open(f"./mmseg_train/splits/fold_{fold}.txt", "w") as f:
        for idx in train_idx:
            f.write(os.path.basename(all_image_files[idx])[:-4] + "\n")
    with open(f"./mmseg_train/splits/holdout_{fold}.txt", "w") as f:
        for idx in valid_idx:
            f.write(os.path.basename(all_image_files[idx])[:-4] + "\n")

os.system("mkdir -p ./mmseg_train/splits_notail ./mmseg_train/splits_noanno ./mmseg_train/splits_case")
tails = set()
noannos = set()
faults = set()
for day, group in df_train.groupby("days"):
    end_slice = group.slice.iloc[np.where(~group.segmentation.isna())[0][-1]]
    tail_slice = np.arange(end_slice + 1, end_slice + 6)
    tail_group = group[group.slice.isin(tail_slice)].drop_duplicates(["days", "image_files"])
    noanno_group = group[group.segmentation.isna()].drop_duplicates(["days", "image_files"])
    tails.update([os.path.basename(row.image_files)[:-4].replace("slice", row.days) for i, row in tail_group.iterrows()])
    noannos.update([os.path.basename(row.image_files)[:-4].replace("slice", row.days) for i, row in noanno_group.iterrows()])
    if day in ['case7_day0', 'case81_day30']:
        faults.update([os.path.basename(row.image_files)[:-4].replace("slice", row.days) for i, row in group.iterrows()])
    
for f in range(5):
    split = pd.read_csv(f"./mmseg_train/splits/fold_{f}.txt", header = None)
    x = list(set(split.iloc[:,0].tolist()) - tails)
    pd.DataFrame(x).to_csv(f"./mmseg_train/splits_notail/fold_{f}.txt", index = False, header = False)
    x = list(set(split.iloc[:,0].tolist()) - noannos)
    pd.DataFrame(x).to_csv(f"./mmseg_train/splits_noanno/fold_{f}.txt", index = False, header = False)
    x = list(set(split.iloc[:,0].tolist()) - faults)
    pd.DataFrame(x).to_csv(f"./mmseg_train/splits_case/fold_{f}.txt", index = False, header = False)
    
    
    split = pd.read_csv(f"./mmseg_train/splits/holdout_{f}.txt", header = None)
    x = list(set(split.iloc[:,0].tolist()) - tails)
    pd.DataFrame(x).to_csv(f"./mmseg_train/splits_notail/holdout_{f}.txt", index = False, header = False)
    x = list(set(split.iloc[:,0].tolist()) - noannos)
    pd.DataFrame(x).to_csv(f"./mmseg_train/splits_noanno/holdout_{f}.txt", index = False, header = False)
    x = list(set(split.iloc[:,0].tolist()) - faults)
    pd.DataFrame(x).to_csv(f"./mmseg_train/splits_case/holdout_{f}.txt", index = False, header = False)

    for d in ["", "_notail", "_noanno", "_case"]:
        os.system(f"cat ./mmseg_train/splits{d}/holdout_{f}.txt > ./mmseg_train/splits{d}/fold_all.txt")
        os.system(f"cat ./mmseg_train/splits{d}/fold_{f}.txt >> ./mmseg_train/splits{d}/fold_all.txt")