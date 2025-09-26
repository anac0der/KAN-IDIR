from utils import general, load_utils, eval_utils
import torch
from models import models_3d
import numpy as np
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
    raise Exception('Wrong model name!')


curr_config = getattr(cfg_module, variable_name)
print(curr_config)

acdc_path = "./data/acdc"
case_ids = range(101, 151)
patient_nums_0 = [[case_id, 0] for case_id in case_ids]
patient_nums_1 = [[case_id, 1] for case_id in case_ids]
patient_nums = patient_nums_0 + patient_nums_1

overall_across_all = dict()
overall_regularity = []
overall_dice_arr = []
overall_runtime = []
overall_hd95 = []
preprocess_as_in_corr_mlp = True # enable preprocessing aligned with CorrMLP
for run in range(args.runs):
    now = datetime.now()
    seed = now.microsecond
    print(f'--- Run {run}, seed = {seed} ---')
    all_mean_dices = []
    
    for pair in patient_nums:
        patient_id = pair[0]
        order = pair[1]
        moving_img, fixed_img, moving_labels, fixed_labels, voxel_size = \
                load_utils.load_pair_acdc(acdc_path, patient_id, order)
        
        if preprocess_as_in_corr_mlp:
            moving_img, moving_labels = general.preprocess_fn_acdc(moving_img, 
                                                                   moving_labels, 
                                                                   voxel_size)
            fixed_img, fixed_labels = general.preprocess_fn_acdc(fixed_img, 
                                                                 fixed_labels, 
                                                                 voxel_size)
        mask_exp = moving_labels.astype('int') + 10 # use all voxels for sampling
        mask_exp = mask_exp.astype('bool').astype('int')
        kwargs = copy.deepcopy(curr_config)
        kwargs["mask"] = mask_exp 
        kwargs['seed'] = seed
        ImpReg = models_3d.ImplicitRegistrator3d(torch.FloatTensor(moving_img),  
                                            torch.FloatTensor(fixed_img), 
                                            **kwargs)
        
        t = time.time()
        ImpReg.fit()
        overall_runtime.append(time.time() - t)

        spacing = (1.5, 1.5, 3.15) if preprocess_as_in_corr_mlp else voxel_size
        mean_dice, dice_arr, hd95 = (
            eval_utils.eval_segmentation_accuracy(
            ImpReg, 
            fixed_img.shape, 
            fixed_labels, 
            moving_labels, 
            voxel_size=spacing, 
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
        order_str = 'ES -> ED' if order else 'ED -> ES'
        print("Patient {}, {},  mean: {:.3f}, std: {:.3f}, HD95: {:.3f}, folded voxels %: {:.5f}".format(
            patient_id, order_str, mean_dice, np.std(dice_arr), hd95, folded_voxels_percent
        ))
        
        pair_key = tuple(pair)
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