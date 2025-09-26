import numpy as np
import SimpleITK as sitk
import nibabel as nib
import torch
import os

def load_image_DIRLab(variation=1, folder="D:\Data\DIRLAB\Case"):
    image_sizes = [
        0,
        [94, 256, 256],
        [112, 256, 256],
        [104, 256, 256],
        [99, 256, 256],
        [106, 256, 256],
        [128, 512, 512],
        [136, 512, 512],
        [128, 512, 512],
        [128, 512, 512],
        [120, 512, 512],
    ]

    voxel_sizes = [
        0,
        [2.5, 0.97, 0.97],
        [2.5, 1.16, 1.16],
        [2.5, 1.15, 1.15],
        [2.5, 1.13, 1.13],
        [2.5, 1.1, 1.1],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
    ]

    shape = image_sizes[variation]

    folder = folder + str(variation) + "Pack" + os.path.sep

    # Images
    dtype = np.dtype(np.int16)

    with open(folder + "Images/case" + str(variation) + "_T00_s.img", "rb") as f:
        data = np.fromfile(f, dtype)
    image_insp = data.reshape(shape)

    with open(folder + "Images/case" + str(variation) + "_T50_s.img", "rb") as f:
        data = np.fromfile(f, dtype)
    image_exp = data.reshape(shape)

    imgsitk_in = sitk.ReadImage(folder + "Masks/case" + str(variation) + "_T00_s.mhd")

    mask = np.clip(sitk.GetArrayFromImage(imgsitk_in), 0, 1)

    image_insp = torch.FloatTensor(image_insp)
    image_exp = torch.FloatTensor(image_exp)

    # Landmarks
    with open(
        folder + "ExtremePhases/Case" + str(variation) + "_300_T00_xyz.txt"
    ) as f:
        landmarks_insp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    with open(
        folder + "ExtremePhases/Case" + str(variation) + "_300_T50_xyz.txt"
    ) as f:
        landmarks_exp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    landmarks_insp[:, [0, 2]] = landmarks_insp[:, [2, 0]]
    landmarks_exp[:, [0, 2]] = landmarks_exp[:, [2, 0]]

    return (
        image_insp,
        image_exp,
        landmarks_insp,
        landmarks_exp,
        mask,
        voxel_sizes[variation],
    )


def load_pair_oasis_3d(dataset_path, folders, i, j):
    def load_image(i):
        path = os.path.join(dataset_path, folders[i])
        image_path = os.path.join(path, 'aligned_norm.nii.gz')
        nim1 = nib.load(image_path)
        image1 = nim1.get_fdata()
        image1 = np.array(image1, dtype='float32')

        return image1
    
    def load_label(i):
        path = os.path.join(dataset_path, folders[i])
        image_path = os.path.join(path, 'aligned_seg35.nii.gz')
        nim1 = nib.load(image_path)
        label1 = nim1.get_fdata()
        label1 = np.array(label1, dtype='float32')
        return label1
    
    return load_image(i), load_image(j), load_label(i), load_label(j)


def load_pair_acdc(dataset_path, patient_num, order=0):
    metadata_path = os.path.join(dataset_path, f'patient{patient_num}', f'Info.cfg')
    i = 0
    ed_num = 0
    es_num = 0
    with open(metadata_path, 'r') as f:
        for line in f:
            idx = line.strip().split(' ')[-1]
            if i == 0:
                ed_num = idx
            else:
                es_num = idx
            i += 1
            if i == 2:
                break
        
    es_path = os.path.join(dataset_path, 
                           f'patient{patient_num}', 
                           f'patient{patient_num}_frame{es_num.zfill(2)}.nii')
                           
    ed_path = os.path.join(dataset_path, f'patient{patient_num}', 
                           f'patient{patient_num}_frame{ed_num.zfill(2)}.nii')
    
    es_img = nib.load(es_path)
    header = es_img.header
    es_img = es_img.get_fdata()
    voxel_size = header.get_zooms()
    es_img = np.array(es_img, dtype=np.float32)
    ed_img = nib.load(ed_path).get_fdata()
    ed_img = np.array(ed_img, dtype=np.float32)

    es_gt_path = es_path.replace('.nii', '_gt.nii')
    ed_gt_path = ed_path.replace('.nii', '_gt.nii')
    es_gt = nib.load(es_gt_path).get_fdata()
    es_gt = np.array(es_gt, dtype=np.float32)
    ed_gt = nib.load(ed_gt_path).get_fdata()
    ed_gt = np.array(ed_gt, dtype=np.float32)

    if order:
        fixed_img = ed_img
        moving_img = es_img
        fixed_labels = ed_gt
        moving_labels = es_gt
    else:
        fixed_img = es_img
        moving_img = ed_img
        fixed_labels = es_gt
        moving_labels = ed_gt
    return moving_img, fixed_img, moving_labels, fixed_labels, voxel_size