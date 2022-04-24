# Kaggle

## UW-Madison GI Tract Image Segmentation

### submission log

id|model|cv score|lb score| model  |batch size|iters|loss|image size|aug|trick
--|-----|--------|--------|----|------|--|--|--|--|--
0|convx_t_16x2_baseline_f0|77.76|0.848|convxt|16x2|20k|ce|384|flip rot90 color|
1|convx_t_16x2_10k_512_f0|77.30||convxt|16x2|10k|ce|512|flip rot90 color|
2|convx_t_16x2_15k_dcp_aug0_f0|77.61||convxt|16x2|15k|ce+0.5dice|384|color|
2.1|convx_b_16x2_15k_dcp_aug0_f0|77.61||convxb|16x2|15k|ce+0.5dice|384|color|
3|convx_t_16x2_15k_dice_f0|76.94||convxt|16x2|15k|0.5ce+0.5dice|384|flip rot90 color|
4|convx_t_16x2_15k_diceplus_f0|77.52||convxt|16x2|15k|ce+0.5dice|384|flip rot90 color|
5|convx_t_32x2_15k_diceplus_f0|77.55||convxt|32x2|15k|ce+0.5dice|384|flip rot90 color|
6|convx_t_16x2_20k_aug_3img_f0|78.69||convxt|16x2|20k|ce|384|flip rot90 color|3img
7|convx_t_16x2_20k_dcp_aug0_3img_f0|78.79||convxt|16x2|15k|ce+0.5dice|384|color|3img
8|convx_t_16x2_20k_aug2_3img_f0|78.54||convxt|16x2|20k|ce|384| flip rot90 color cutout |3img
9|convx_t_16x2_20k_dcp_aug0_3img_f0|79.28|0.849|convxt|16x2|20k|ce+0.5dice|384|color|3img
10|convx_t_16x2_20k_dcp_aug1_3img_f0|79.41||connxt|16x2|20k|ce+0.5dice|384|color cutout|3img