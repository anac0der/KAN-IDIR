from utils import general, load_utils, eval_utils
import torch
from models import models_3d
import numpy as np
import os
import importlib
import argparse
import copy
from datetime import datetime
import time

torch.set_num_threads(16)
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='configs.config')
parser.add_argument('--model', type=str, default='kan')
parser.add_argument('--runs', type=int, default=1)

args = parser.parse_args()
cfg_module = importlib.import_module(args.config_path)
if args.model is not None:
    variable_name = args.model.lower()
else:
    raise Exception('EWrong model name!')


curr_config = getattr(cfg_module, variable_name)
print(curr_config)

oasis_path = "./data/oasis_1_3d"
oasis_folders_path = os.path.join(oasis_path, "subjects.txt")

folders = []
with open(oasis_folders_path, 'r') as f:
    for line in f:
        folders.append(line.strip('\n'))

case_ids = range(364, 413)
pair_list = [[i, i + 1] for i in case_ids]

print(len(pair_list))

overall_across_all = dict()
overall_regularity = []
overall_dice_arr = []
overall_runtime = []
overall_hd95 = []
for run in range(args.runs):
    now = datetime.now()
    seed = now.microsecond
    print(f'--- RUn {run}, seed = {seed} ---')
    all_mean_dices = []
    
    for pair in pair_list:
        moving_id, fixed_id = pair[0], pair[1]

        moving_img, fixed_img, moving_labels, fixed_labels = (
            load_utils.load_pair_oasis_3d(oasis_path, 
                                          folders, 
                                          moving_id, 
                                          fixed_id)
        )

        mask_exp = moving_labels.astype('int')
        mask_exp = mask_exp.astype('bool').astype('int')
        kwargs = copy.deepcopy(curr_config)
        kwargs["mask"] = mask_exp 
        kwargs['seed'] = seed
        ImpReg = models_3d.ImplicitRegistrator3d(torch.FloatTensor(moving_img),  
                                            torch.FloatTensor(fixed_img), 
                                            **kwargs)
        
        torch.cuda.empty_cache()  # Clear cache before starting
        t = time.time()
        ImpReg.fit()
        overall_runtime.append(time.time() - t)

        mean_dice, dice_arr, hd95 = (
            eval_utils.eval_segmentation_accuracy(ImpReg, 
                                                  fixed_img.shape, 
                                                  fixed_labels, 
                                                  moving_labels, 
                                                  device='cuda')
        )

        folded_voxels_percent = eval_utils.compute_deformation_regularity(
            ImpReg.network, 
            ImpReg.possible_coordinate_tensor, 
            output_shape=fixed_img.shape
        )
        
        overall_regularity.append(folded_voxels_percent)
        all_mean_dices.append(mean_dice)
        overall_dice_arr.append(dice_arr)
        overall_hd95.append(hd95)

        print("Pair: {} and {},  " \
              "mean: {:.3f}, std: {:.3f}, HD95: {:.3f}, folded voxels %: {:.5f}".format(
            moving_id, fixed_id, mean_dice, np.std(dice_arr), hd95, folded_voxels_percent
        ))
        
        pair_key = (moving_id, fixed_id)
        if pair_key in overall_across_all.keys():
            overall_across_all[pair_key].append(mean_dice)
        else:
            overall_across_all[pair_key] = [mean_dice]

    print(f'Overall with seed = {seed}: {np.array(all_mean_dices).mean():.3f}')

print(f'Cfg: {curr_config}')
print(f'---- Results after {args.runs} runs ----')
final_mean = 0
for pair_key in overall_across_all.keys():
    pair_mean = np.array(overall_across_all[pair_key]).mean()
    final_mean += pair_mean

final_dice_mean = final_mean / len(overall_across_all)
final_dice_std = np.concatenate(overall_dice_arr, axis=0).std()
print(f"Overall dice, mean: {final_dice_mean:.3f}")
print(f"Overall dice, std: {final_dice_std:.3f}")
hd95_overall = np.array(overall_hd95).mean()
hd95_overall_std = np.array(overall_hd95).std()
print(f"Overall HD95, mean: {hd95_overall:.3f}")
print(f"Overall HD95, std: {hd95_overall_std:.3f}")
folded_vox_percent_overall = np.array(overall_regularity).mean()
folded_vox_percent_overall_std = np.array(overall_regularity).std()
print(f'% of folded voxels, mean: {folded_vox_percent_overall:.5f}')
print(f'% of folded voxels, std: {folded_vox_percent_overall_std:.5f}')
print("Mean runtime: {:.2f} seconds".format(np.array(overall_runtime).mean()))