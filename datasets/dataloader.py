import os
import random
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomGrayscale
import torchvision.transforms.functional as F


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area


class SemiCrowdDataset(torch.utils.data.Dataset):

    def __init__(self, labeled_file_list, labeled_main_transform=None, labeled_img_transform=None,
                 labeled_dmap_transform=None, labeled_attmap_transform=None, phase='train'):

        self.labeled_data_files = []
        with open(labeled_file_list, 'r')as f:
            lines = f.readlines()
            for line in lines:
                self.labeled_data_files.append(line.strip())
        f.close()

        self.phase = phase
        # if self.phase == 'train':
        #     self.labeled_data_files = self.labeled_data_files * 8

        self.label_main_transform = labeled_main_transform
        self.label_img_transform = labeled_img_transform
        self.label_dmap_transform = labeled_dmap_transform
        self.labeled_attmap_transform = labeled_attmap_transform

    def __len__(self):
        return len(self.labeled_data_files)

    def __getitem__(self, index):
        index = index % len(self.labeled_data_files)
        labeled_image_filename = self.labeled_data_files[index]
        labeled_gt_filename = labeled_image_filename.replace('.jpg', '_densitymap.npy')

        img = Image.open(labeled_image_filename)
        if img.mode == 'L':
            img = img.convert('RGB')
        # dmap = Image.open(labeled_gt_filename)
        dmap = np.load(labeled_gt_filename)
        dmap = dmap.astype(np.float32, copy=False)
        dmap = Image.fromarray(dmap)

        attmap = None
        keypoints = None

        if self.label_main_transform is not None:
            img, dmap, keypoints, attmap = self.label_main_transform((img, dmap, keypoints, attmap))
        if self.label_img_transform is not None:
            img = self.label_img_transform(img)
        if self.label_dmap_transform is not None:
            dmap = self.label_dmap_transform(dmap)
        if self.labeled_attmap_transform is not None:
            attmap = self.labeled_attmap_transform(attmap) if attmap is not None else None

        return {'image': img,
                'densitymap': dmap,
                'imagepath': labeled_image_filename}



class SemiUnlabelCrowdDataset(torch.utils.data.Dataset):

    def __init__(self, unlabeled_file_list, unlabeled_transform):
        self.unlabeled_data_files = []

        with open(unlabeled_file_list, 'r')as f:
            lines = f.readlines()
            for line in lines:
                self.unlabeled_data_files.append(line.strip())
        f.close()

        self.transform = unlabeled_transform

    def __len__(self):
        return len(self.unlabeled_data_files)

    def __getitem__(self, index):

        index = index % len(self.unlabeled_data_files)
        filename = self.unlabeled_data_files[index]

        img = Image.open(filename)
        if img.mode == 'L':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return {'image': img, 'imagepath': filename}


def create_semi_labeled_train_dataloader(labeled_file_list, use_flip, batch_size, cropsize=256, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    main_trans_list = []
    if use_flip:
        main_trans_list.append(RandomHorizontalFlip())
    main_trans_list.append(RandomCrop(cropsize=cropsize))

    main_trans = Compose(main_trans_list)
    img_trans = Compose([RandomGrayscale(p=0.2), ToTensor(), Normalize(mean=mean, std=std)])
    dmap_trans = ToTensor()
    attmap_trans = ToTensor()

    dataset = SemiCrowdDataset(labeled_file_list=labeled_file_list, labeled_main_transform=main_trans,
                               labeled_img_transform=img_trans, labeled_dmap_transform=dmap_trans,
                               labeled_attmap_transform=attmap_trans, phase='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader


def create_semi_unlabel_train_dataloader(unlabeled_file_list, batch_size, use_flip=True, cropsize=256, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    main_trans_list = []
    if use_flip:
        main_trans_list.append(transforms.RandomHorizontalFlip())
    main_trans_list.append(RandomCrop(cropsize=cropsize))
    main_trans_list.append(RandomGrayscale(p=0.2))
    main_trans_list.append(ToTensor())
    main_trans_list.append(Normalize(mean=mean, std=std))
    main_trans = Compose(main_trans_list)

    dataset = SemiUnlabelCrowdDataset(unlabeled_file_list=unlabeled_file_list, unlabeled_transform=main_trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader


def create_semi_val_or_test_dataloader(file_list):
    main_trans_list = []
    main_trans = Compose(main_trans_list)
    img_trans = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dmap_trans = ToTensor()
    attmap_trans = ToTensor()

    dataset = SemiCrowdDataset(labeled_file_list=file_list, labeled_main_transform=main_trans,
                               labeled_img_transform=img_trans, labeled_dmap_transform=dmap_trans,
                               labeled_attmap_transform=attmap_trans, phase='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    return dataloader


class RandomHorizontalFlip(object):
    '''
    Random horizontal flip.
    prob = 0.5
    '''

    def __call__(self, img_and_dmap):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap, keypoints, attmap = img_and_dmap
        if attmap is None:
            if random.random() < 0.5:
                if keypoints is not None:
                    keypoints[:, 0] = img.size[0] - keypoints[:, 0]
                    if keypoints[:,0].min()<0:
                        print('keypoints[:,0].min()<0')
                    return (img.transpose(Image.FLIP_LEFT_RIGHT), dmap.transpose(Image.FLIP_LEFT_RIGHT), keypoints, attmap)
                else:
                    return (img.transpose(Image.FLIP_LEFT_RIGHT), dmap.transpose(Image.FLIP_LEFT_RIGHT), keypoints, attmap)
            else:
                return (img, dmap, keypoints, attmap)

        else:
            if random.random() < 0.5:
                if keypoints is not None:
                    keypoints[:, 0] = img.size[0] - keypoints[:, 0]
                    if keypoints[:,0].min()<0:
                        print('keypoints[:,0].min()<0')
                    return (img.transpose(Image.FLIP_LEFT_RIGHT), dmap.transpose(Image.FLIP_LEFT_RIGHT), keypoints, attmap.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    return (img.transpose(Image.FLIP_LEFT_RIGHT), dmap.transpose(Image.FLIP_LEFT_RIGHT), keypoints, attmap.transpose(Image.FLIP_LEFT_RIGHT))
            else:
                return (img, dmap, keypoints, attmap)


class RandomResize(object):
    '''
    Random resize.
    '''

    def __init__(self, factor=16):
        self.factor = factor

    @staticmethod
    def get_params(img, factor):
        w, h = img.size
        if w % factor == 0 and h % factor == 0:
            return 0, 0, h, w
        else:
            return 0, 0, h - (h % factor), w - (w % factor)

    def __call__(self, img):
        '''
        img: PIL.Image
        '''

        # img = img.resize((1024, 768), Image.ANTIALIAS)

        i, j, th, tw = self.get_params(img, self.factor)

        img = F.crop(img, i, j, th, tw)
        return img


class PairedCrop(object):
    '''
    Paired Crop for both image and its density map.
    Note that due to the maxpooling in the nerual network,
    we must promise that the size of input image is the corresponding factor.
    '''

    def __init__(self, factor=8):
        self.factor = factor

    @staticmethod
    def get_params(img, factor):
        w, h = img.size
        if w % factor == 0 and h % factor == 0:
            return 0, 0, h, w
        else:
            return 0, 0, h - (h % factor), w - (w % factor)

    def __call__(self, img_and_dmap):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap, keypoints, attmap = img_and_dmap

        i, j, th, tw = self.get_params(img, self.factor)

        img = F.crop(img, i, j, th, tw)
        dmap = F.crop(dmap, i, j, th, tw)
        attmap = F.crop(attmap, i, j, th, tw) if attmap is not None else None

        return (img, dmap, keypoints, attmap)


class RandomCrop(object):
    def __init__(self, cropsize=256):
        self.size = (cropsize, cropsize)

    @staticmethod
    def get_params(img, size):
        w, h = img.size
        th, tw = size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h-th+1, size=(1,)).item()
        j = torch.randint(0, w-tw+1, size=(1,)).item()

        return i,j,th,tw

    def __call__(self, img_and_dmap):
        # self.size = (img.size[1] // 2, img.size[0] // 2)
        # self.size = (400, 400)
        # self.size = (256, 256)

        if isinstance(img_and_dmap, tuple):
            img, dmap, keypoints, attmap = img_and_dmap

            i, j, h, w = self.get_params(img, self.size)
            img = F.crop(img, i, j, h, w)
            dmap = F.crop(dmap, i, j, h, w)

            return (img, dmap, keypoints, attmap)
        else:
            img = img_and_dmap

            i, j, h, w = self.get_params(img, self.size)
            img = F.crop(img, i, j, h, w)

            return img

