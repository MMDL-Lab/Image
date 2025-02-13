import logging
import os
import numpy as np
import random
from tqdm import tqdm
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2

from models.initializer import initialize_from_cfg
import matplotlib.pyplot as plt
logger = logging.getLogger("global_logger")

__all__ = [
    "GRAD",
]

def compute_con_loss(query_abnormal, block_mask, th=0.3, device='cuda:0'):
    B1, C1, H1, W1 = query_abnormal.shape
    num_anomaly = torch.sum(block_mask)
    block_mask = block_mask.to(device)
    # Repeat block_mask along the first dimension B1 times
    block_mask = block_mask.repeat(B1, 1, 1, 1)

    # Flatten the tensors for easier manipulation
    query_abnormal_flat = query_abnormal[:, 0, :, :].view(B1, -1)
    block_mask_flat = block_mask[:, 0, :, :].view(B1, -1)

    # Compute the losses using vectorized operations
    con_loss_a = torch.sum(torch.clamp(th - query_abnormal_flat, min=0) * block_mask_flat)
    con_loss_n = torch.sum(torch.clamp(th + query_abnormal_flat, min=0) * (1 - block_mask_flat))

    # Compute the final loss
    con_loss = con_loss_a / (num_anomaly + 1e-8) + con_loss_n / (B1 * H1 * W1 - num_anomaly)

    return con_loss

def conv3x3(inplanes, outplanes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        inplanes,
        outplanes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(inplanes, outplanes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class GRAD(nn.Module):
    def __init__(
        self,
        block,
        layers,
        frozen_layers=[],
        groups=1,
        width_per_group=64,
        norm_layer=None,
        initializer=None,
        **kwargs
    ):
        super(GRAD, self).__init__()
        block = globals()[block]
        
        ch = kwargs['inplanes'][0]
        coord_ch = kwargs['inplanes'][0]
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 2*ch
        self.dilation = 1
        self.frozen_layers = frozen_layers
        self.groups = groups
        self.base_width = width_per_group
        h, w = kwargs['feature_size']
        self.save_recon = kwargs['save_recon']
        self.mse_lamb = kwargs['mse_lamb']
        self.cos_lamb = kwargs['cos_lamb']
        self.mse_coef = kwargs['mse_coef']
        self.noise_std = kwargs['noise_std']

        # normal grid
        self.query_normal = nn.ParameterList([
            nn.Parameter(nn.init.xavier_normal_(torch.empty(1, ch, kwargs['local_resol'], kwargs['local_resol']))), 
            nn.Parameter(nn.init.xavier_normal_(torch.empty(1, ch*h*w, kwargs['global_resol'], kwargs['global_resol'])))
        ])
        # anomalous grid
        self.query_abnormal = nn.ParameterList([
            nn.Parameter(nn.init.xavier_normal_(torch.empty(1, ch, kwargs['local_resol'], kwargs['local_resol']))), 
            nn.Parameter(nn.init.xavier_normal_(torch.empty(1, ch*h*w, kwargs['global_resol'], kwargs['global_resol'])))
        ])

        # global
        self.coord_ = nn.Sequential(
            nn.Linear(coord_ch, coord_ch), 
            nn.Tanh(), 
            nn.Linear(coord_ch,2), 
            nn.Tanh()
        )

        # local
        self.coord = nn.Sequential(
            nn.Conv2d(coord_ch,coord_ch,1,1,0), 
            nn.Tanh(), 
            nn.Conv2d(coord_ch,2,1,1,0), 
            nn.Tanh()
        )

        self.layer_normal = nn.Sequential(
            self._make_layer(block, 2*ch, layers[0]),
        )

        self.layer_abnormal = nn.Sequential(
            self._make_layer(block, 2*ch, layers[0]),
        )

        self.recover_normal = conv1x1(2*ch, coord_ch)\
        
        self.recover_abnormal = conv1x1(2*ch, coord_ch)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)

        self.weight_normal = nn.Parameter(torch.tensor(-0.2, dtype=torch.float32))

        initialize_from_cfg(self, initializer)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)
 
    def random_walk_mask(self, size, start_x, start_y, max_steps=100):
        mask = torch.zeros(size, size)
        x, y = start_x, start_y
        mask[x, y] = 1
        for _ in range(max_steps):
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            x, y = x + dx, y + dy
            if 0 <= x < size and 0 <= y < size:
                mask[x, y] = 1
        return mask
  
    def forward(self, input):
        feature_align = input["feature_align"]
        B, C, H, W = feature_align.shape
        block_mask = torch.zeros(torch.Size([1, 1, H, W]))

        # FBP
        if input['mode'] == "train_abnormal" and self.training:
            block_size = int(random.uniform(0.5, 10))
            block_intensity = random.uniform(8.0, 16.0) 
            block_center_x = random.randint(block_size, feature_align.size(2) - block_size - 1)
            block_center_y = random.randint(block_size, feature_align.size(3) - block_size - 1)
            block_paste = torch.zeros(torch.Size([1, 1, H, W])).cuda()

            mask = self.random_walk_mask(2 * block_size + 1, block_size, block_size)
            for i in range(-block_size, block_size + 1):
                for j in range(-block_size, block_size + 1):
                    if block_center_x + i >= 0 and block_center_x + i < feature_align.size(2) and \
                    block_center_y + j >= 0 and block_center_y + j < feature_align.size(3):
                        if mask[block_size + i, block_size + j] == 1:
                            block_paste[..., block_center_x + i, block_center_y + j] += block_intensity
                            block_mask[..., block_center_x + i, block_center_y + j] += 1
            feature_align += F2.gaussian_blur(block_paste, [3, 3])

        # Coordinate Mapping
        coord_ = self.coord_(nn.functional.adaptive_avg_pool2d(feature_align, (1, 1)).view(B,-1)) # global
        coord = self.coord(feature_align) # local

        # Coordinate jitter
        add_noise = True if torch.rand(1).item() < 0.5 else False
        if self.training and add_noise: coord = coord + torch.randn(B,2,H,W, device=coord.device)*self.noise_std

        # When training one grid, fix another one
        if input['mode'] == "train_abnormal" and self.training:
            for param in self.query_normal.parameters():
                param.requires_grad = False
            for param in self.query_abnormal.parameters():
                param.requires_grad = True    
        elif input['mode'] == "train_normal" and self.training:
            for param in self.query_normal.parameters():
                param.requires_grad = True
            for param in self.query_abnormal.parameters():
                param.requires_grad = False
        else:
            for param in self.query_normal.parameters():
                param.requires_grad = False
            for param in self.query_abnormal.parameters():
                param.requires_grad = False

        # Normal grid sampling
        query_normal = torch.cat([
            F.grid_sample(self.query_normal[1], coord_.view(1,B,1,2), align_corners=False).permute(2,3,0,1).view(B,-1,H,W), # global
            F.grid_sample(self.query_normal[0].repeat(B,1,1,1), coord.permute(0,2,3,1), align_corners=False) # local
        ], dim=1)

        # Anomalous grid sampling
        query_abnormal = torch.cat([
            F.grid_sample(self.query_abnormal[1], coord_.view(1,B,1,2), align_corners=False).permute(2,3,0,1).view(B,-1,H,W), # global
            F.grid_sample(self.query_abnormal[0].repeat(B,1,1,1), coord.permute(0,2,3,1), align_corners=False) # local
        ], dim=1)

        mean = torch.mean(query_abnormal, dim=(2, 3), keepdim=True)
        std = torch.std(query_abnormal, dim=(2, 3), keepdim=True)
        query_abnormal = (query_abnormal - mean) / std

        # Processing through Convolutional Blocks
        feature_rec = self.layer_normal(query_normal) 
        feature_rec = self.recover_normal(feature_rec)
        query_abnormal = self.layer_abnormal(query_abnormal)
        query_abnormal = self.recover_abnormal(query_abnormal)

        # Calculate Contrastive Loss
        th = 0.5
        con_loss = 0
        if input['mode'] == "train_abnormal" and self.training:
            con_loss = compute_con_loss(query_abnormal, block_mask, th)

        # fuse
        feature_rec += self.weight_normal * query_abnormal

        # Feature refinement
        mse = torch.mean((feature_rec - feature_align)**2, dim=1) 
        mse = 1 - torch.round((mse*self.mse_coef).clamp(0,1))
        cos = F.cosine_similarity(feature_rec, feature_align, dim=1)
        sim = (self.mse_lamb*mse + self.cos_lamb*cos).unsqueeze(1)
        feature_rec = sim*feature_align + (1-sim)*feature_rec 
        

        # Post processing of prediction results
        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align)**2, dim=1, keepdim=True)
        )
        pred = self.upsample(pred)

        return {
            'feature_rec': feature_rec,
            'feature_align': feature_align,
            'pred': pred,
            'con_loss':con_loss
        }
