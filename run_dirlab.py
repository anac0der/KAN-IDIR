from utils import general, load_utils, eval_utils
from models import models_3d
import numpy as np
import torch
import importlib
import argparse
import copy
from datetime import datetime
import time

data_dir = "./data/dirlab"
out_dir = "output"

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
case_ids = range(1, 11)
overall_across_all = dict()
overall_diffs = dict()
overall_regularity = []
overall_runtime = []
for run in range(args.runs):
    now = datetime.now()
    seed = now.microsecond
    print(f'--- Run {run}, seed = {seed} ---')
    overall = 0
    for case_id in case_ids:

        (
            img_insp,
            img_exp,
            landmarks_insp,
            landmarks_exp,
            mask_exp,
            voxel_size,
        ) = load_utils.load_image_DIRLab(case_id, "{}/Case".format(data_dir))

        kwargs = copy.deepcopy(curr_config)
        kwargs["mask"] = mask_exp  
        kwargs['seed'] = seed
        torch.cuda.empty_cache() 

        t = time.time()
        ImpReg = models_3d.ImplicitRegistrator3d(img_exp, img_insp, **kwargs)
        ImpReg.fit()
        overall_runtime.append(time.time() - t)
        new_landmarks_orig, delta = eval_utils.compute_landmarks(
           ImpReg.network, landmarks_insp, image_size=img_insp.shape
        ) 

        accuracy_mean, accuracy_std, all_accuracies = (
            eval_utils.compute_landmark_accuracy(
                new_landmarks_orig, 
                landmarks_exp, 
                voxel_size=voxel_size
            )
        )
        
        folded_voxels_percent = eval_utils.compute_deformation_regularity(
            ImpReg.network, 
            ImpReg.possible_coordinate_tensor, 
            output_shape=img_insp.shape
        )

        overall_regularity.append(folded_voxels_percent)
        print("Case id: {} mean: {} std: {} folded % {:.5f}".format(case_id, 
                                                                    accuracy_mean[0], 
                                                                    accuracy_std[0], 
                                                                    folded_voxels_percent))
        overall += accuracy_mean[0]
        if case_id in overall_across_all.keys():
            overall_across_all[case_id].append(accuracy_mean[0])
            overall_diffs[case_id].append(all_accuracies)
        else:
            overall_across_all[case_id] = [accuracy_mean[0]]
            overall_diffs[case_id] = [all_accuracies]
    print(f'Overall with seed = {seed}: {overall / len(case_ids)}')
    
print(f'Cfg: {curr_config}')
print(f'---- Results after {args.runs} runs ----')
final_mean = 0
final_accuracies = []
for key in overall_across_all.keys():
    case_mean = np.array(overall_across_all[key]).mean()
    final_case_accs = np.concatenate(overall_diffs[key])
    case_std = np.std(final_case_accs)
    print(f'Case id: {key}, mean: {case_mean:.2f}, std: {case_std:.2f}, ')
    final_mean += case_mean
    final_accuracies.append(final_case_accs)

final_mean = final_mean / len(case_ids)
final_std = np.std(np.concatenate(final_accuracies))
print(f'Overall mean: {final_mean :.3f}, overall std: {final_std:.2f}')
folded_vox_percent_overall = np.array(overall_regularity).mean()
folded_vox_percent_overall_std = np.array(overall_regularity).std()
print(f'% of folded voxels, mean: {folded_vox_percent_overall:.5f}')
print(f'% of folded voxels, std: {folded_vox_percent_overall_std:.5f}')
print("Mean runtime: {:.2f}".format(np.array(overall_runtime).mean()))