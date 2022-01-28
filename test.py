import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision


from datasets.dataloader import create_semi_val_or_test_dataloader
from models.CSRNet import CSRNet_SEMI_L2R, CSRNet_SEMI_TwoStage, CSRNet_SEMI_Multistage, CSRNet_SEMI

import numpy as np
import time
import os
import sys
import errno
import argparse
import math
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test crowdcounting model')

parser.add_argument('--dataset', type=str, default='shanghaitech')
parser.add_argument('--test-files', type=str, help='your test file')
parser.add_argument('--best-model', type=str, help='your pretrained model path')
parser.add_argument('--use-avai-gpus', action='store_true')
parser.add_argument('--gpu-devices', type=str, default='0')
parser.add_argument('--model', type=str, default='CSRNet_SEMI')
parser.add_argument('--test-batch', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)


args = parser.parse_args()


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

criterion = nn.MSELoss(reduction='sum')

if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
use_gpu = torch.cuda.is_available()

if use_gpu:
    print("Currently using GPU {}".format(args.gpu_devices))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
else:
    print("Currently using CPU (GPU is highly recommended)")


test_loader = create_semi_val_or_test_dataloader(args.test_files)

model = CSRNet_SEMI().cuda()


if os.path.isfile(args.best_model):
    print('loading checkpoints: ', args.best_model)
    pkl = torch.load(args.best_model)
    state_dict = pkl['state_dict']
    print("Currently epoch {}".format(pkl['epoch']))

    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})


model.eval()

with torch.no_grad():
    epoch_mae = 0.0
    epoch_rmse_loss = 0.0
    for i, data in enumerate(tqdm(test_loader)):
        image = data['image'].cuda()
        gt_densitymap = data['densitymap'].cuda()
        et_densitymap = model(image).detach()
        print('prediction: ', str(et_densitymap.sum()))
        print('gt: ', str(gt_densitymap.sum()))

        mae = abs(et_densitymap.data.sum() - gt_densitymap.sum())
        rmse = mae * mae

        epoch_mae += mae.item()
        epoch_rmse_loss += rmse.item()

    epoch_mae /= len(test_loader.dataset)
    epoch_rmse_loss = math.sqrt(epoch_rmse_loss / len(test_loader.dataset))
print("bestmae: ", epoch_mae)
print("rmse: ", epoch_rmse_loss)

