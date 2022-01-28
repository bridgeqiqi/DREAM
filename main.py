import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import math

from datasets.dataloader import create_semi_labeled_train_dataloader, create_semi_unlabel_train_dataloader, create_semi_val_or_test_dataloader
from optims.optimizers import init_optim

from models.CSRNet import CSRNet_SEMI, CSRNet_SEMI_Multistage, CSRNet_SEMI_L2R, CSRNet_SEMI_TwoStage

from losses.MarginRankLoss import MixedLoss

from utils.pytorch_utils import AverageMeter

import numpy as np
import time
import os
import sys
import errno
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Train crowdcounting model')

parser.add_argument('--dataset', type=str, default='shanghaitech')
parser.add_argument('--unlabeldataset', type=str, default='CAPTURE')
parser.add_argument('--label-file-list', type=str)
parser.add_argument('--unlabel-file-list', type=str)
parser.add_argument('--val-file-list', type=str)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--use-avai-gpus', action='store_true')
parser.add_argument('--gpu-devices', type=str, default='0')
parser.add_argument('--model', type=str, default='ResNet')
parser.add_argument('--max-epoch', type=int, default=10)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1e-05)
parser.add_argument('--weight-decay', default=1e-04, type=float)
parser.add_argument('--train-batch', type=int)
parser.add_argument('--train-unlabel-batch', type=int)
parser.add_argument('--val-batch', type=int, default=1)
parser.add_argument('--label-crop-size', type=int, default=256)
parser.add_argument('--unlabel-crop-size', type=int, default=256)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--loss', type=str, default='mixedloss')
parser.add_argument('--semi', action='store_true')
parser.add_argument('--crop-ratio', type=float, default=0.75)
parser.add_argument('--lamda', type=float, default=1.0)
parser.add_argument('--label-percent', type=str, default='100')


parser.add_argument('--checkpoints', type=str, default='./checkpoints')
parser.add_argument('--summary-writer', type=str, default='./runs')
parser.add_argument('--save-txt', type=str, default='train_log.txt')

args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def main():
    SAVE_CHECKPOINT_SUBROOT = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.dataset, args.unlabeldataset, args.model, args.label_percent, args.optim, args.lr,
        args.weight_decay, args.train_batch, args.train_unlabel_batch,
        args.label_crop_size, args.unlabel_crop_size, args.lamda
    )
    args.checkpoints = os.path.join(args.checkpoints, SAVE_CHECKPOINT_SUBROOT)
    args.summary_writer = os.path.join(args.summary_writer, SAVE_CHECKPOINT_SUBROOT)
    mkdir_if_missing(args.checkpoints)
    mkdir_if_missing(args.summary_writer)
    SEMI = args.semi = True  # set True manually for debug
    print("learning rate: ", args.lr)
    print("dataset: ", args.dataset)
    print("model: ", args.model)
    print("labelfilelist: ", args.label_file_list)
    print("unlabelfilelist: ", args.unlabel_file_list)
    print("valfilelist: ", args.val_file_list)
    print("train batch: ", args.train_batch)
    print("train unlabel batch: ", args.train_unlabel_batch)
    print("label crop size: ", args.label_crop_size)
    print("unlabel crop size: ", args.unlabel_crop_size)
    print("val batch: ", args.val_batch)
    print("optimizer: ", args.optim)
    print("weight decay: ", args.weight_decay)
    print("IsSemi: ", SEMI)
    print("Loss function: ", args.loss)
    print("Lambda: ", args.lamda)
    print("Label_Percent: ", args.label_percent)
    print("checkpoints: ", args.checkpoints)
    print("runs: ", args.summary_writer)



    torch.manual_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")


    BATCH_NORMALIZATION = False
    if args.train_batch > 1:
        BATCH_NORMALIZATION = True
    print("Batch Normalization: ", BATCH_NORMALIZATION)

    if args.model == 'CSRNet':
        model = CSRNet_SEMI()
    elif args.model == 'CSRNet_Multi':
        model = CSRNet_SEMI_Multistage()
    elif args.model == 'CSRNet_L2R':
        model = CSRNet_SEMI_L2R()
    elif args.model == 'CSRNet_TwoStage':
        model = CSRNet_SEMI_TwoStage()

    print("Currently using {} model".format(args.model))
    print("The parameters of model are: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    start_epoch = 0
    if args.resume == '':
        if os.path.isfile(args.resume):
            pkl = torch.load(args.resume)
            state_dict = pkl['state_dict']
            print("Currently epoch {}".format(pkl['epoch']))
            model.load_state_dict(state_dict)
            start_epoch = pkl['epoch']
    print("Currently epoch {}".format(start_epoch))

    if args.loss == 'mixedloss':
        total_criterion = MixedLoss(lamda=args.lamda)

    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)


    label_train_loader = create_semi_labeled_train_dataloader(labeled_file_list=args.label_file_list,
                                                              use_flip=True,
                                                              batch_size=args.train_batch,
                                                              cropsize=args.label_crop_size)
    unlabel_train_loader = create_semi_unlabel_train_dataloader(unlabeled_file_list=args.unlabel_file_list,
                                                                use_flip=True,
                                                                batch_size=args.train_unlabel_batch,
                                                                cropsize=args.unlabel_crop_size)
    val_loader = create_semi_val_or_test_dataloader(file_list=args.val_file_list)


    min_mae = sys.maxsize
    min_mae_epoch = -1

    writer = SummaryWriter(args.summary_writer)

    COUNT = 0

    with open(os.path.join(args.checkpoints, args.save_txt), 'a') as f:
        for epoch in range(start_epoch, start_epoch + args.max_epoch):
            model.train()
            epoch_loss = AverageMeter()
            epoch_label_loss = AverageMeter()
            epoch_unlabel_loss = AverageMeter()
            epoch_mae = AverageMeter()
            epoch_mse = AverageMeter()

            for i, label_data in enumerate(tqdm(label_train_loader)):
                epoch_start = time.time()
                label_image = label_data['image'].cuda()
                gt_densitymap = label_data['densitymap'].cuda()
                N = label_image.shape[0]

                for j, unlabel_data in enumerate(unlabel_train_loader):
                    unlabel_image = unlabel_data['image'].cuda()
                    break

                label_et_densitymap, unlabel_et_densitymap = model(label_image, unlabel_image)

                losses = total_criterion(label_et_densitymap, gt_densitymap, unlabel_et_densitymap)

                epoch_loss.update(losses['total'].item(), N)
                epoch_label_loss.update(losses['label'].item(), N)
                if isinstance(losses['unlabel'], float):
                    epoch_unlabel_loss.update(losses['unlabel'], N)
                else:
                    epoch_unlabel_loss.update(losses['unlabel'].item(), N)

                loss = losses['total']

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                pred_count = torch.sum(label_et_densitymap.view(N, -1), dim=1).detach().cpu().numpy()
                gt_count = torch.sum(gt_densitymap.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gt_count
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

                epoch_end = time.time()


            print('Epoch {} Train, Loss: {:.6f}, LabelLoss: {:.6f}, UnlabelLoss: {:.6f}, '
                  'MAE: {:.2f}, RMSE:{:.2f}, Cost: {:.1f} sec'
                  .format(epoch, epoch_loss.get_avg(), epoch_label_loss.get_avg(),
                          epoch_unlabel_loss.get_avg(), epoch_mae.get_avg(),
                          np.sqrt(epoch_mse.get_avg()), epoch_end-epoch_start))

            f.write('Epoch {} Train, Loss: {:.6f}, LabelLoss: {:.6f}, UnlabelLoss: {:.6f}, '
                  'MAE: {:.2f}, RMSE:{:.2f}, Cost: {:.1f} sec'
                  .format(epoch, epoch_loss.get_avg(), epoch_label_loss.get_avg(),
                          epoch_unlabel_loss.get_avg(), epoch_mae.get_avg(),
                          np.sqrt(epoch_mse.get_avg()), epoch_end-epoch_start) + "\n")

            writer.add_scalar('Train_Total_Loss', epoch_loss.get_avg(), epoch)
            writer.add_scalar('Train_Label_Loss', epoch_label_loss.get_avg(), epoch)
            writer.add_scalar('Train_Unlabel_Loss', epoch_unlabel_loss.get_avg(), epoch)
            writer.add_scalar('Train_MAE', epoch_mae.get_avg(), epoch)
            writer.add_scalar('Train_RMSE', np.sqrt(epoch_mse.get_avg()), epoch)
            writer.add_scalar('Train_Time', epoch_end-epoch_start, epoch)


            model.eval()
            epoch_mae = 0.0
            epoch_rmse = 0.0
            with torch.no_grad():

                for i, data in enumerate(tqdm(val_loader)):
                    epoch_start = time.time()
                    image = data['image'].cuda()
                    gt_densitymap = data['densitymap'].cuda()
                    et_densitymap = model(image).detach()

                    mae = abs(et_densitymap.data.sum() - gt_densitymap.sum())
                    rmse = mae * mae

                    epoch_mae += mae.item()
                    epoch_rmse += rmse.item()
                    epoch_end = time.time()


                epoch_mae /= len(val_loader.dataset)
                epoch_rmse = math.sqrt(epoch_rmse / len(val_loader.dataset))

                print('Epoch {} Val, MAE: {:.2f} RMSE: {:.2f}, Cost {:.1f} sec'
                      .format(epoch, epoch_mae, epoch_rmse, epoch_end-epoch_start))
                f.write('Epoch {} Val, MAE: {:.2f} RMSE: {:.2f}, Cost {:.1f} sec'
                      .format(epoch, epoch_mae, epoch_rmse, epoch_end-epoch_start) + "\n")

                if epoch_mae <= min_mae:
                    min_mae, min_rmse, min_mae_epoch = epoch_mae, epoch_rmse, epoch
                    torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_mae': min_mae,
                        'best_rmse': min_rmse
                    }, os.path.join(args.checkpoints, "bestmodel_{}.pth".format(COUNT)))
                    COUNT += 1


                writer.add_scalar('Val_MAE', epoch_mae, epoch)
                writer.add_scalar('Val_RMSE', epoch_rmse, epoch)
                writer.add_scalar('Val_Time', epoch_end-epoch_start, epoch)
                writer.add_scalar('Val_BEST_MAE', min_mae, epoch)
                print('Epoch {} Val, BEST_MAE: {:.2f} BEST_RMSE: {:.2f}'
                      .format(epoch, min_mae, min_rmse))
                f.write('Epoch {} Val, MAE: {:.2f} RMSE: {:.2f}'
                        .format(epoch, min_mae, min_rmse) + "\n")

    f.close()
    return


if __name__ == '__main__':
    main()