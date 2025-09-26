import numpy as np
from medpy.metric.binary import hd95

def compute_dice(pred1, truth1):
    mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    dice_list = []
    for k in mask_value4[1:]:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        dice_list.append(intersection / (np.sum(pred) + np.sum(truth)))
    return np.mean(dice_list), dice_list

def compute_hd95(pred1, truth1, spacing):
    mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    hd95_list = []
    for k in mask_value4[1:]:
        truth = truth1 == k
        pred = pred1 == k
        hd95_list.append(hd95(pred, truth, voxelspacing=spacing))
    return np.mean(hd95_list)