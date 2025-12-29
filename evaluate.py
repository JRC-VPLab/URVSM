import os
from os.path import exists
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np

from utils.dataloader import Retinal_loader_eval
from utils.metrics import cal_acc, specificity, sensitivity, precision, F1_SCORE, MCC_SCORE, AUC_SCORE, dice_score, clDice_score, getBettiNumber
from models.cyclegan import define_cyclegan_G
from models.segmentation import UNet_vanilla, ResDO_UNet

parser = argparse.ArgumentParser()

# Essential
parser.add_argument('--datapath', default='./data/JRCFA', help='path to original images')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--note', type=str, default='test_dataset_name')

# model
parser.add_argument('--translation_model', type=str, default='CycleGAN', choices=['CycleGAN'])
parser.add_argument('--seg_model', default='unet_vanilla', type=str, choices=['unet_vanilla', 'Resdounet'])
parser.add_argument('--transnet_checkpoint', default='./ckpt/translation.pth', help='path to translation model checkpoint')
parser.add_argument('--segnet_checkpoint_unet', default='./ckpt/segmentation.pth', help='path to segmentation model checkpoint')
parser.add_argument('--segnet_checkpoint_resdounet', default='./ckpt/segmentation_resdo.pth', help='path to segmentation model checkpoint')

parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--netG', type=str, default='unet_128v', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', type=bool, default=True, help='no dropout for the generator')

# test
parser.add_argument('--binarize_thres', type=float, default=0.4, help='threshold < 0.5 is recommended since the model tends to underpredict')
parser.add_argument('--compute_Betti_number', type=bool, default=False, help='Skip Betti number computation can significantly improve the speed')

# save
parser.add_argument('--save_translation', default='./result', help='path to translation model checkpoint')
parser.add_argument('--save_segmentation', default='./result', help='path to translation model checkpoint')

class MetricRecorder():
    def __init__(self, Betti_number=False):
        super().__init__()
        self.metrics = defaultdict(list)
        self.Betti_number = Betti_number

    def update(self, pred, gt):
        '''
        Input:
            pred: (H, W), value in {0, 1}, torch.float32
            gt: (H, W), value in {0, 1}, torch.float32
        '''

        acc = cal_acc(pred, gt)
        dice = dice_score(pred, gt)
        cldice = clDice_score(pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))
        f1 = F1_SCORE(pred, gt)
        sp = specificity(pred, gt)
        se = sensitivity(pred, gt)
        if self.Betti_number:
            B0_P, B1_P, B0_T, B1_T = getBettiNumber(pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))

        self.metrics['accuracy'].append(acc)
        self.metrics['dice'].append(dice)
        self.metrics['cldice'].append(cldice)
        self.metrics['f1'].append(f1)
        self.metrics['sp'].append(sp)
        self.metrics['se'].append(se)
        if self.Betti_number:
            self.metrics['B0'].append(np.abs(B0_P - B0_T))
            self.metrics['B1'].append(np.abs(B1_P - B1_T))

    def show(self):
        for metric_name, vals in self.metrics.items():
            print(metric_name, np.mean(vals))


def main():
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')

    args.save_translation = os.path.join(args.save_translation, args.note, 'translation')
    if not exists(args.save_translation):
        os.makedirs(args.save_translation)
    args.save_segmentation = os.path.join(args.save_segmentation, args.note, 'segmentation')
    if not exists(args.save_segmentation):
        os.makedirs(args.save_segmentation)

    # Dataloader
    eval_dataset = Retinal_loader_eval(os.path.join(args.datapath, 'img'), os.path.join(args.datapath, 'label'))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    # model
    # image translation model
    if args.translation_model == 'CycleGAN':
        trans_net = define_cyclegan_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.norm, not args.no_dropout, args.init_type, args.init_gain, args.gpu_id)
        trans_net.load_state_dict(torch.load(args.transnet_checkpoint))
    else:
        raise ValueError('Specified image translation backbone unavailable.')

    # vessel segmentation model
    if args.seg_model == 'unet_vanilla':
        seg_net = UNet_vanilla()
        seg_net.load_state_dict(torch.load(args.segnet_checkpoint_unet))
    elif args.seg_model == 'Resdounet':
        seg_net = ResDO_UNet(in_ch=4, out_ch=1)
        seg_net.load_state_dict(torch.load(args.segnet_checkpoint_resdounet))
    else:
        raise ValueError('Specified segmentation backbone unavailable.')

    trans_net = trans_net.to(args.device)
    seg_net = seg_net.to(args.device)

    metrics = MetricRecorder(Betti_number=args.compute_Betti_number)
    metrics_nt = MetricRecorder(Betti_number=args.compute_Betti_number)

    with torch.no_grad():
        for i, (sample, gt, name) in enumerate(eval_loader):
            print(name[0])

            # ==============  Image Translation  ==============
            sample = sample.permute(0, 3, 1, 2).to(torch.float32).to(args.device)
            N, C, H, W = sample.size()
            assert N == 1  # use batch size 1

            fake_sample = trans_net(sample)

            # postprocessing and save
            fake_sample = torch.clamp(fake_sample, min=0.0, max=1.0)
            fake_sample_save = (fake_sample.squeeze(0).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)  # (N, C, H, W) -> (H, W, C)
            fake_sample_save = cv.cvtColor(fake_sample_save, cv.COLOR_BGR2RGB)
            cv.imwrite(os.path.join(args.save_translation, '{}.png'.format(name[0])), fake_sample_save)

            # ==============  Image Segmentation  ==============
            init_seg = torch.zeros([N, 1, H, W], dtype=torch.float32).to(args.device)
            init_seg_nt = torch.zeros([N, 1, H, W], dtype=torch.float32).to(args.device)
            net_in = torch.cat((init_seg, fake_sample), dim=1)  # input: 3-channel RGB image + 1-channel blank segmentation map
            net_in_nt = torch.cat((init_seg_nt, sample), dim=1)

            y = seg_net(net_in)
            y_nt = seg_net(net_in_nt)

            for i in range(1):
                net_in = torch.cat((y, fake_sample), dim=1)
                net_in_nt = torch.cat((y_nt, sample), dim=1)
                y = seg_net(net_in)
                y_nt = seg_net(net_in_nt)

            # save segmentation results
            out = y[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
            out_nt = y_nt[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
            cv.imwrite(os.path.join(args.save_segmentation, '{}.png'.format(name[0])), (out * 255).astype(np.uint8))
            # no translation, using topological-aware segmentation model only (segmentation model still trained with translated samples)
            cv.imwrite(os.path.join(args.save_segmentation, '{}_nt.png'.format(name[0])), (out_nt * 255).astype(np.uint8))

            # post-processing for metric computation
            gt = gt[0].to(torch.float32).cpu()
            h, w = gt.shape
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
            y_nt = F.interpolate(y_nt, size=(h, w), mode='bilinear', align_corners=False)

            y = y[0, 0, :, :].cpu()  # (H, W)
            y_nt = y_nt[0, 0, :, :].cpu()  # (H, W

            # binarize prediction
            y = (y > args.binarize_thres).to(torch.float32)
            y_nt = (y_nt > args.binarize_thres).to(torch.float32)

            # compute metrics
            metrics.update(y, gt)
            metrics_nt.update(y_nt, gt)

        print('Full model:')
        metrics.show()
        print('\nTopological segmentation model only:')
        metrics_nt.show()


if __name__ == '__main__':
    main()