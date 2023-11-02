import torch.nn.functional as F
import torch
import torch.nn as nn


def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


def regMSE(vec_field):
    return torch.mean(vec_field[:,:,:,0]**2 + vec_field[:,:,:,1]**2)


def regSampleCoords(grid_to_sample):
    # return torch.mean(torch.clamp(torch.abs(grid_to_sample) - 1, min=0)**2)
    return torch.mean(grid_to_sample**2)



def log_regression_loss(disp_est, disp_gt, mask, ensemble_gt, pred_uncertainty):
    # construct 0-1 gt
    disp_est = disp_est * mask.float()
    gt = torch.abs(disp_est - disp_gt)

    bce_loss = nn.BCELoss()
    for i in range(gt.shape[0]):
        temp = gt[i, :, :]
        temp[temp>=0] = 1
        
        # hard prior supervision
        temp[temp<3] = 1
        temp[temp>=3] = 0

        # soft prior supervision
        ensemble_temp = ensemble_gt[i, :, :]
        ensemble_temp[ensemble_temp > 10] = 10
        ensemble_temp = 1-ensemble_temp/10

        temp = ensemble_temp * temp
        

        temp = temp * mask[i,:,:].float()
        gt[i, :, :] = temp

    loss = bce_loss(pred_uncertainty[mask], gt[mask])

    return loss

