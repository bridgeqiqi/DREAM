import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections

import numpy as np
import cv2
import random


class CSRNet_SEMI(nn.Module):
    '''
    output size: 1/8
    '''

    def __init__(self, batch_norm=False, load_weights=False, crop_ratio=0.75):
        super(CSRNet_SEMI, self).__init__()
        self.crop_ratio = crop_ratio
        self.frontend_feat = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512]
        self.frontend = make_layers(self.frontend_feat, batch_norm=batch_norm)
        self.backend_feat = [512,512,512,256,128,64]
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True, batch_norm=batch_norm)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            if batch_norm:
                mod = torchvision.models.vgg16_bn(pretrained=True)
            else:
                mod = torchvision.models.vgg16(pretrained=True)
            # self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, label_x, unlabel_x=None):
        if self.training:
            # labeled image processing
            label_x = self.frontend(label_x)
            label_x = self.backend(label_x)
            label_x = F.interpolate(label_x, scale_factor=8, mode='bilinear')
            label_x = self.output_layer(label_x)

            # unlabeled image processing
            B,C,H,W = unlabel_x.shape
            unlabel_x = self.frontend(unlabel_x)

            unlabel_x = self.generate_feature_patches(unlabel_x, self.crop_ratio)
            assert unlabel_x.shape[0] == B*5

            unlabel_x = self.backend(unlabel_x)
            unlabel_x = self.output_layer(unlabel_x)
            unlabel_x = torch.split(unlabel_x, split_size_or_sections=B, dim=0)

            return label_x, unlabel_x

        else:

            label_x = self.frontend(label_x)
            label_x = self.backend(label_x)
            label_x = F.interpolate(label_x, scale_factor=8, mode='bilinear')
            label_x = self.output_layer(label_x)

            return label_x

    def generate_feature_patches(self, unlabel_x, ratio=0.75):
        # unlabeled image processing

        unlabel_x_1 = unlabel_x
        b, c, h, w = unlabel_x.shape

        center_x = random.randint(h // 2 - (h - h * ratio) // 2, h // 2 + (h - h * ratio) // 2)
        center_y = random.randint(w // 2 - (w - w * ratio) // 2, w // 2 + (w - w * ratio) // 2)

        new_h2 = int(h * ratio)
        new_w2 = int(w * ratio)  # 48*48
        unlabel_x_2 = unlabel_x[:, :, center_x - new_h2 // 2:center_x + new_h2 // 2,
                      center_y - new_w2 // 2:center_y + new_w2 // 2]

        new_h3 = int(new_h2 * ratio)
        new_w3 = int(new_w2 * ratio)
        unlabel_x_3 = unlabel_x[:, :, center_x - new_h3 // 2:center_x + new_h3 // 2,
                      center_y - new_w3 // 2:center_y + new_w3 // 2]

        new_h4 = int(new_h3 * ratio)
        new_w4 = int(new_w3 * ratio)
        unlabel_x_4 = unlabel_x[:, :, center_x - new_h4 // 2:center_x + new_h4 // 2,
                      center_y - new_w4 // 2:center_y + new_w4 // 2]

        new_h5 = int(new_h4 * ratio)
        new_w5 = int(new_w4 * ratio)
        unlabel_x_5 = unlabel_x[:, :, center_x - new_h5 // 2:center_x + new_h5 // 2,
                      center_y - new_w5 // 2:center_y + new_w5 // 2]

        unlabel_x_2 = nn.functional.interpolate(unlabel_x_2, size=(h, w), mode='bilinear')
        unlabel_x_3 = nn.functional.interpolate(unlabel_x_3, size=(h, w), mode='bilinear')
        unlabel_x_4 = nn.functional.interpolate(unlabel_x_4, size=(h, w), mode='bilinear')
        unlabel_x_5 = nn.functional.interpolate(unlabel_x_5, size=(h, w), mode='bilinear')

        unlabel_x = torch.cat([unlabel_x_1, unlabel_x_2, unlabel_x_3, unlabel_x_4, unlabel_x_5], dim=0)

        return unlabel_x



class CSRNet_SEMI_TwoStage(nn.Module):
    '''
    output size: 1/8
    '''

    def __init__(self, batch_norm=False, load_weights=False, crop_ratio=0.75):
        super(CSRNet_SEMI_TwoStage, self).__init__()
        self.crop_ratio = crop_ratio
        self.frontend_feat = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512]
        self.frontend = make_layers(self.frontend_feat, batch_norm=batch_norm)
        self.backend_feat = [512,512,512,256,128,64]
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True, batch_norm=batch_norm)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.conv1x1_2 = nn.Conv2d(256, 512, kernel_size=1)


        if not load_weights:
            if batch_norm:
                mod = torchvision.models.vgg16_bn(pretrained=True)
            else:
                mod = torchvision.models.vgg16(pretrained=True)
            # self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, label_x, unlabel_x=None):
        if self.training:
            # labeled image processing
            label_x = self.frontend(label_x)
            label_x = self.backend(label_x)
            label_x = F.interpolate(label_x, scale_factor=8, mode='bilinear')
            label_x = self.output_layer(label_x)

            # unlabeled image processing
            B,C,H,W = unlabel_x.shape
            unlabel_x_2 = self.frontend[:16](unlabel_x)
            unlabel_x_3 = self.frontend[16:](unlabel_x_2)

            unlabel_x_2 = self.generate_feature_patches(unlabel_x_2, self.crop_ratio)
            unlabel_x_3 = self.generate_feature_patches(unlabel_x_3, self.crop_ratio)

            assert unlabel_x_2.shape[0] == B * 5
            assert unlabel_x_3.shape[0] == B * 5

            unlabel_x_2 = self.conv1x1_2(unlabel_x_2)

            unlabel_x_2 = self.output_layer(self.backend(unlabel_x_2))
            unlabel_x_3 = self.output_layer(self.backend(unlabel_x_3))

            unlabel_x_2 = torch.split(unlabel_x_2, split_size_or_sections=B, dim=0)
            unlabel_x_3 = torch.split(unlabel_x_3, split_size_or_sections=B, dim=0)

            return label_x, [unlabel_x_2, unlabel_x_3]

        else:

            label_x = self.frontend(label_x)
            label_x = self.backend(label_x)
            label_x = F.interpolate(label_x, scale_factor=8, mode='bilinear')
            label_x = self.output_layer(label_x)

            return label_x


    def generate_feature_patches(self, unlabel_x, ratio=0.75):
        # unlabeled image processing

        unlabel_x_1 = unlabel_x
        b, c, h, w = unlabel_x.shape

        center_x = random.randint(h // 2 - (h - h * ratio) // 2, h // 2 + (h - h * ratio) // 2)
        center_y = random.randint(w // 2 - (w - w * ratio) // 2, w // 2 + (w - w * ratio) // 2)

        new_h2 = int(h * ratio)
        new_w2 = int(w * ratio)  # 48*48
        unlabel_x_2 = unlabel_x[:, :, center_x - new_h2 // 2:center_x + new_h2 // 2,
                      center_y - new_w2 // 2:center_y + new_w2 // 2]

        new_h3 = int(new_h2 * ratio)
        new_w3 = int(new_w2 * ratio)
        unlabel_x_3 = unlabel_x[:, :, center_x - new_h3 // 2:center_x + new_h3 // 2,
                      center_y - new_w3 // 2:center_y + new_w3 // 2]

        new_h4 = int(new_h3 * ratio)
        new_w4 = int(new_w3 * ratio)
        unlabel_x_4 = unlabel_x[:, :, center_x - new_h4 // 2:center_x + new_h4 // 2,
                      center_y - new_w4 // 2:center_y + new_w4 // 2]

        new_h5 = int(new_h4 * ratio)
        new_w5 = int(new_w4 * ratio)
        unlabel_x_5 = unlabel_x[:, :, center_x - new_h5 // 2:center_x + new_h5 // 2,
                      center_y - new_w5 // 2:center_y + new_w5 // 2]

        unlabel_x_2 = nn.functional.interpolate(unlabel_x_2, size=(h, w), mode='bilinear')
        unlabel_x_3 = nn.functional.interpolate(unlabel_x_3, size=(h, w), mode='bilinear')
        unlabel_x_4 = nn.functional.interpolate(unlabel_x_4, size=(h, w), mode='bilinear')
        unlabel_x_5 = nn.functional.interpolate(unlabel_x_5, size=(h, w), mode='bilinear')

        unlabel_x = torch.cat([unlabel_x_1, unlabel_x_2, unlabel_x_3, unlabel_x_4, unlabel_x_5], dim=0)

        return unlabel_x



class CSRNet_SEMI_Multistage(nn.Module):
    '''
    output size: 1/8
    '''

    def __init__(self, batch_norm=False, load_weights=False, crop_ratio=0.75):
        super(CSRNet_SEMI_Multistage, self).__init__()
        self.crop_ratio = crop_ratio
        self.frontend_feat = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512]
        self.frontend = make_layers(self.frontend_feat, batch_norm=batch_norm)
        self.backend_feat = [512,512,512,256,128,64]
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True, batch_norm=batch_norm)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.conv1x1_1 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(256, 512, kernel_size=1)


        if not load_weights:
            if batch_norm:
                mod = torchvision.models.vgg16_bn(pretrained=True)
            else:
                mod = torchvision.models.vgg16(pretrained=True)
            # self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, label_x, unlabel_x=None):
        if self.training:
            # labeled image processing
            label_x = self.frontend(label_x)
            label_x = self.backend(label_x)
            label_x = F.interpolate(label_x, scale_factor=8, mode='bilinear')
            label_x = self.output_layer(label_x)

            # unlabeled image processing
            B,C,H,W = unlabel_x.shape
            unlabel_x_1 = self.frontend[:9](unlabel_x)
            unlabel_x_2 = self.frontend[9:16](unlabel_x_1)
            unlabel_x_3 = self.frontend[16:](unlabel_x_2)

            unlabel_x_1 = self.generate_feature_patches(unlabel_x_1, self.crop_ratio)
            unlabel_x_2 = self.generate_feature_patches(unlabel_x_2, self.crop_ratio)
            unlabel_x_3 = self.generate_feature_patches(unlabel_x_3, self.crop_ratio)

            assert unlabel_x_1.shape[0] == B * 5
            assert unlabel_x_2.shape[0] == B * 5
            assert unlabel_x_3.shape[0] == B * 5

            unlabel_x_1 = self.conv1x1_1(unlabel_x_1)
            unlabel_x_2 = self.conv1x1_2(unlabel_x_2)


            unlabel_x_1 = self.output_layer(self.backend(unlabel_x_1))
            unlabel_x_2 = self.output_layer(self.backend(unlabel_x_2))
            unlabel_x_3 = self.output_layer(self.backend(unlabel_x_3))

            unlabel_x_1 = torch.split(unlabel_x_1, split_size_or_sections=B, dim=0)
            unlabel_x_2 = torch.split(unlabel_x_2, split_size_or_sections=B, dim=0)
            unlabel_x_3 = torch.split(unlabel_x_3, split_size_or_sections=B, dim=0)

            return label_x, [unlabel_x_1, unlabel_x_2, unlabel_x_3]

        else:

            label_x = self.frontend(label_x)
            label_x = self.backend(label_x)
            label_x = F.interpolate(label_x, scale_factor=8, mode='bilinear')
            label_x = self.output_layer(label_x)

            return label_x

    def generate_feature_patches(self, unlabel_x, ratio=0.75):
        # unlabeled image processing

        unlabel_x_1 = unlabel_x
        b, c, h, w = unlabel_x.shape

        center_x = random.randint(h // 2 - (h - h * ratio) // 2, h // 2 + (h - h * ratio) // 2)
        center_y = random.randint(w // 2 - (w - w * ratio) // 2, w // 2 + (w - w * ratio) // 2)

        new_h2 = int(h * ratio)
        new_w2 = int(w * ratio)  # 48*48
        unlabel_x_2 = unlabel_x[:, :, center_x - new_h2 // 2:center_x + new_h2 // 2,
                      center_y - new_w2 // 2:center_y + new_w2 // 2]

        new_h3 = int(new_h2 * ratio)
        new_w3 = int(new_w2 * ratio)
        unlabel_x_3 = unlabel_x[:, :, center_x - new_h3 // 2:center_x + new_h3 // 2,
                      center_y - new_w3 // 2:center_y + new_w3 // 2]

        new_h4 = int(new_h3 * ratio)
        new_w4 = int(new_w3 * ratio)
        unlabel_x_4 = unlabel_x[:, :, center_x - new_h4 // 2:center_x + new_h4 // 2,
                      center_y - new_w4 // 2:center_y + new_w4 // 2]

        new_h5 = int(new_h4 * ratio)
        new_w5 = int(new_w4 * ratio)
        unlabel_x_5 = unlabel_x[:, :, center_x - new_h5 // 2:center_x + new_h5 // 2,
                      center_y - new_w5 // 2:center_y + new_w5 // 2]

        unlabel_x_2 = nn.functional.interpolate(unlabel_x_2, size=(h, w), mode='bilinear')
        unlabel_x_3 = nn.functional.interpolate(unlabel_x_3, size=(h, w), mode='bilinear')
        unlabel_x_4 = nn.functional.interpolate(unlabel_x_4, size=(h, w), mode='bilinear')
        unlabel_x_5 = nn.functional.interpolate(unlabel_x_5, size=(h, w), mode='bilinear')

        unlabel_x = torch.cat([unlabel_x_1, unlabel_x_2, unlabel_x_3, unlabel_x_4, unlabel_x_5], dim=0)

        return unlabel_x


def make_layers(layer_list, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for layer in layer_list:
        if layer == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif layer == 'U':
            layers.append(nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1))
            in_channels = in_channels // 2
        else:
            conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=d_rate, dilation=d_rate)

            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=False)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=False)])

            in_channels = layer

    return nn.Sequential(*layers)

class CSRNet_SEMI_L2R(nn.Module):
    '''
    output size: 1/8
    '''

    def __init__(self, batch_norm=False, load_weights=False, crop_ratio=0.75):
        super(CSRNet_SEMI_L2R, self).__init__()
        self.crop_ratio = crop_ratio
        self.frontend_feat = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512]
        self.frontend = make_layers(self.frontend_feat, batch_norm=batch_norm)
        self.backend_feat = [512,512,512,256,128,64]
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True, batch_norm=batch_norm)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            if batch_norm:
                mod = torchvision.models.vgg16_bn(pretrained=True)
            else:
                mod = torchvision.models.vgg16(pretrained=True)
            # self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, label_x, unlabel_x=None):
        if self.training:
            # labeled image processing
            label_x = self.frontend(label_x)
            label_x = self.backend(label_x)
            label_x = F.interpolate(label_x, scale_factor=8, mode='bilinear')
            label_x = self.output_layer(label_x)

            # unlabeled image processing
            B,C,H,W = unlabel_x.shape

            unlabel_x = self.generate_feature_patches(unlabel_x, self.crop_ratio)
            assert unlabel_x.shape[0] == B*5

            unlabel_x = self.frontend(unlabel_x)
            unlabel_x = self.backend(unlabel_x)

            unlabel_x = self.output_layer(unlabel_x)
            unlabel_x = torch.split(unlabel_x, split_size_or_sections=B, dim=0)

            return label_x, unlabel_x

        else:

            label_x = self.frontend(label_x)
            label_x = self.backend(label_x)
            label_x = F.interpolate(label_x, scale_factor=8, mode='bilinear')
            label_x = self.output_layer(label_x)

            return label_x

    def generate_feature_patches(self, unlabel_x, ratio=0.75):
        # unlabeled image processing

        unlabel_x_1 = unlabel_x
        b, c, h, w = unlabel_x.shape

        center_x = random.randint(h // 2 - (h - h * ratio) // 2, h // 2 + (h - h * ratio) // 2)
        center_y = random.randint(w // 2 - (w - w * ratio) // 2, w // 2 + (w - w * ratio) // 2)

        new_h2 = int(h * ratio)
        new_w2 = int(w * ratio)  # 48*48
        unlabel_x_2 = unlabel_x[:, :, center_x - new_h2 // 2:center_x + new_h2 // 2,
                      center_y - new_w2 // 2:center_y + new_w2 // 2]

        new_h3 = int(new_h2 * ratio)
        new_w3 = int(new_w2 * ratio)
        unlabel_x_3 = unlabel_x[:, :, center_x - new_h3 // 2:center_x + new_h3 // 2,
                      center_y - new_w3 // 2:center_y + new_w3 // 2]

        new_h4 = int(new_h3 * ratio)
        new_w4 = int(new_w3 * ratio)
        unlabel_x_4 = unlabel_x[:, :, center_x - new_h4 // 2:center_x + new_h4 // 2,
                      center_y - new_w4 // 2:center_y + new_w4 // 2]

        new_h5 = int(new_h4 * ratio)
        new_w5 = int(new_w4 * ratio)
        unlabel_x_5 = unlabel_x[:, :, center_x - new_h5 // 2:center_x + new_h5 // 2,
                      center_y - new_w5 // 2:center_y + new_w5 // 2]

        unlabel_x_2 = nn.functional.interpolate(unlabel_x_2, size=(h, w), mode='bilinear')
        unlabel_x_3 = nn.functional.interpolate(unlabel_x_3, size=(h, w), mode='bilinear')
        unlabel_x_4 = nn.functional.interpolate(unlabel_x_4, size=(h, w), mode='bilinear')
        unlabel_x_5 = nn.functional.interpolate(unlabel_x_5, size=(h, w), mode='bilinear')

        unlabel_x = torch.cat([unlabel_x_1, unlabel_x_2, unlabel_x_3, unlabel_x_4, unlabel_x_5], dim=0)

        return unlabel_x


