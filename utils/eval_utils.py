import numpy as np
import torch
import torch.nn as nn
from .metrics import compute_dice, compute_hd95
from .general import make_coordinate_tensor_3d

NJD_BS = 50000 #Batch size during NJD computation
SEG_BS = 500000 #Batch size during DSC/HD95 computation

def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds, difference

def compute_landmarks(network, landmarks_pre, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]
    coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0

    network.eval()
    output = network(coordinate_tensor.cuda(), 0)
    delta = output.cpu().detach().numpy() * (scale_of_axes)
    return landmarks_pre + delta, delta

def eval_segmentation_accuracy(imp_reg, output_shape, fixed_labels, moving_labels, 
                               voxel_size=(1., 1., 1.,), device='cuda'):  
     
    coords = make_coordinate_tensor_3d(output_shape)
    network = imp_reg.network.to(device)
    network.eval()
    with torch.no_grad():
        output = torch.zeros_like(coords).to('cpu')
        bs = SEG_BS
        i = 0
        N = coords.shape[0]
        while i < N:
            curr_coords = coords[i:min(N, i + bs)].to(device)
            curr_output = network(curr_coords, 0).detach().cpu()
            output[i:min(N, i + bs)] = curr_output
            i += bs
        output = output.view([output_shape[0], output_shape[1], output_shape[2], 3])
        moving_labels = torch.from_numpy(moving_labels).to('cpu')
        moving_labels = moving_labels[None, ...][None, ...].float()
        _, warped_labels = interp_full_grid_3d(moving_labels,
                                               output[None, ...].float(), 
                                               mod='nearest')
        warped_labels = warped_labels.squeeze(dim=[0, 1])
        wl = warped_labels.detach().long().cpu().numpy()
        mean_dice, dice_arr = compute_dice(wl, fixed_labels.astype('int64'))
        hd95 = compute_hd95(wl, fixed_labels.astype('int64'), voxel_size)
    return mean_dice, dice_arr, hd95
  
def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad

def compute_jacobian_matrix_3d(input_coords, output, add_identity=True):
    """Compute the Jacobian matrix of the output wrt the input."""

    jacobian_matrix = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        jacobian_matrix[:, i, :] = gradient(input_coords, output[:, i])
        if add_identity:
            jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])
    return jacobian_matrix

def compute_deformation_regularity(network, masked_coords, 
                                   output_shape=None, device='cuda'):
    
    network.eval()
    network.to(device)
    masked_coords = make_coordinate_tensor_3d(output_shape)
    bs = NJD_BS
    i = 0
    N = masked_coords.shape[0]
    jacobian_matrix = torch.zeros((N, 3, 3)).to('cpu')
    while i < N:
        curr_coords = masked_coords[i:min(N, i + bs)]
        curr_coords.requires_grad_(True)
        curr_output = network(curr_coords.to(device), 0)
        curr_jac = compute_jacobian_matrix_3d(curr_coords, 
                                              curr_output, 
                                              add_identity=True)
        curr_jac = curr_jac.detach().cpu()
        jacobian_matrix[i:min(N, i + bs)] = curr_jac
        torch.cuda.empty_cache()
        i += bs
    jac_det = torch.det(jacobian_matrix)
    n_folded_voxels = (jac_det < 0).sum().item()
    n_all_voxels = jac_det.shape[0]
    folded_voxels_ratio = n_folded_voxels / n_all_voxels
    return 100 * folded_voxels_ratio

def interp_full_grid_3d(mov_image, flow, mod = 'bilinear'):
    d2, h2, w2 = mov_image.shape[-3:]
    grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), 
                                             torch.linspace(-1, 1, h2), 
                                             torch.linspace(-1, 1, w2)])
    grid_h = grid_h.to(flow.device).float()
    grid_d = grid_d.to(flow.device).float()
    grid_w = grid_w.to(flow.device).float()
    grid_d = nn.Parameter(grid_d, requires_grad=False)
    grid_w = nn.Parameter(grid_w, requires_grad=False)
    grid_h = nn.Parameter(grid_h, requires_grad=False)
    flow_d = flow[:,:,:,:,0]
    flow_h = flow[:,:,:,:,1]
    flow_w = flow[:,:,:,:,2]
    
    disp_d = (grid_d + (flow_d)).squeeze(1)
    disp_h = (grid_h + (flow_h)).squeeze(1)
    disp_w = (grid_w + (flow_w)).squeeze(1)
    sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)
    warped = torch.nn.functional.grid_sample(mov_image, 
                                             sample_grid, 
                                             mode = mod, 
                                             align_corners = True)
        
    return sample_grid, warped