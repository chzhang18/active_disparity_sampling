from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
import gc

from models.guided_stereo import GuidedStereoMatching
from models.gwcnet import GwcNet_GC
from models.adjointnet import AdjointNet


def test_sample(sample, models, args, device):
    #import pdb; pdb.set_trace()
    ad_model, dispnet, guidedstereo = models
    ad_model.eval()
    
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    with torch.no_grad():

        
        pred_disps = dispnet(imgL, imgR)
        pred_disp = pred_disps[0]
        
        pred_disp = torch.unsqueeze(pred_disp, 1) # [B, 1, H, W]

        adjoint_input = torch.cat( (imgL, pred_disp), 1)
        pred_uncertainty = ad_model(adjoint_input)

        pred_uncertainty = pred_uncertainty.cpu().detach().numpy()
        #import pdb; pdb.set_trace()
        pred_uncertainty = 1.0 - pred_uncertainty
        pred_hints = generate_gradient_probability_mask(pred_uncertainty, disp_gt, 30000)
        pred_hints = pred_hints.to(device) * disp_gt
        
        pred_hints = torch.squeeze(pred_hints, 1)
        validhints = (pred_hints > 0).float()

        disp_ests = guidedstereo(imgL, imgR, pred_hints, validhints, k = 10, c = 1)
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        loss = model_loss(disp_ests, disp_gt, mask)
        

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if True:
        with torch.no_grad():
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs)



if __name__ == '__main__':

    cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
    parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

    parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--datapath', required=True, help='data path')
    parser.add_argument('--trainlist', required=True, help='training list')
    parser.add_argument('--testlist', required=True, help='testing list')

    parser.add_argument('--lr', type=float, default=1e-5, help='base learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
    parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

    parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
    parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
    parser.add_argument('--resume', action='store_true', help='continue training the model')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
    parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

    parser.add_argument('-sH', '--samples_height', default=246, type=int, help='number of sparse samples in grid along height dim') # 246
    parser.add_argument('-sW', '--samples_width', default=800, type=int, help='number of sparse samples in grid along width dim') # 800
    parser.add_argument('--ret_samples', action='store_true', help='return the number of samples in the image')
    parser.add_argument('-gpu', type=int, default=2, help='gpu to run on')

    parser.add_argument('--dispnet_kitti', type=str, default='./checkpoints/kitti12/dispnet/' + 'checkpoint_best.ckpt', 
                        help='path to stereo matching model')
    parser.add_argument('--guidedstereo_kitti', type=str, default='./checkpoints/kitti12/guided_stereo/' + 'checkpoint_best.ckpt',
                        help='path to guided stereo matching on kitti data')
    parser.add_argument('--active_sample_kitti', type=str, default='./checkpoints/kitti12/active_sampling/' + 'checkpoint_best.ckpt',
                        help='path to active disparity sampling on kitti data')

    

    # parse arguments, set seeds
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    print("creating model and optimizer")
    # the Adjoint Network
    ad_model = AdjointNet(ngf=64, input_nc=4, output_nc=1)
    ad_model = nn.DataParallel(ad_model)
    ad_model.cuda()
    checkpoint = torch.load(args.active_sample_kitti)
    ad_model.load_state_dict(checkpoint['model'])
    ad_model.eval()

    # optimizer
    optimizer = torch.optim.Adam(ad_model.parameters(), lr=args.lr)


    print("loading dispnet and guided stereo models")
    dispnet = __models__[args.model](args.maxdisp)
    dispnet = nn.DataParallel(dispnet)
    dispnet.cuda()
    checkpoint = torch.load(args.dispnet_kitti)
    dispnet.load_state_dict(checkpoint['model'])
    dispnet.eval()

    # guided stereo model
    guidedstereo = GuidedStereoMatching(args.maxdisp)
    guidedstereo = nn.DataParallel(guidedstereo)
    guidedstereo.cuda()
    checkpoint = torch.load(args.guidedstereo_kitti)
    guidedstereo.load_state_dict(checkpoint['model'])
    guidedstereo.eval()


    models = (ad_model, dispnet, guidedstereo)


    print("creating data loaders")
    # dataset, dataloader
    StereoDataset = __datasets__[args.dataset]
    train_dataset = StereoDataset(args.datapath, args.trainlist, True)
    test_dataset = StereoDataset(args.datapath, args.testlist, False)
    TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)


    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs = test_sample(sample, models, args, device)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss, time.time() - start_time))
    avg_test_scalars = avg_test_scalars.mean()
    print("avg_test_scalars", avg_test_scalars)



