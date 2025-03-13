from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
from deformation import *


class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]

        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        # self._opacity = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.last_size = (self.H, self.W)
        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.m = nn.Parameter(torch.full((self.init_num_points, 1), 1.2))
        self.epsilon = 0.2
        
        # # DeformNetwork
        # self.deform_network = DeformNetwork(D=2, W=256, pos_multires=6, time_multires=6)

            
        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
              
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)


    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound
    
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def forward_dynamic(self, deformed_xyz, deformed_opacity, deformed_features):
        xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(torch.tanh(self._xyz+deformed_xyz), self._cholesky + self.cholesky_bound, self.H, self.W, self.tile_bounds)
        # out_img = rasterize_gaussians_sum(xys, depths, self.radii, conics, num_tiles_hit,
        #         self.get_features+deformed_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = rasterize_gaussians_sum(xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features+deformed_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
                # self.get_features+deformed_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    
    
    def forward_prune(self, deformed_xyz, deformed_opacity, deformed_features):
        soft_mask = torch.sigmoid(self.m)
        binary_mask = (soft_mask > self.epsilon).float()  # determine the ratio to be pruned
        stop_gradient_part = (binary_mask - soft_mask).detach()
        final_mask = stop_gradient_part + soft_mask
        xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(torch.tanh(self._xyz+deformed_xyz), self._cholesky + self.cholesky_bound, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features+deformed_features, self.get_opacity*final_mask, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
                #self.get_features+deformed_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    
    def forward_prune_test(self, deformed_xyz, deformed_opacity, deformed_features):
        soft_mask = torch.sigmoid(self.m)
        binary_mask = (soft_mask > self.epsilon).float() 
        final_mask = binary_mask
        xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(torch.tanh(self._xyz+deformed_xyz), self._cholesky + self.cholesky_bound, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features+deformed_features, self.get_opacity*final_mask, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
                #self.get_features+deformed_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}     
    
    

    def train_iter(self, gt_image, timestamp, stage):
        if stage == "coarse":
            render_pkg = self.forward()
        elif stage == "fine":
            render_pkg = self.forward_dynamic(timestamp)
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()        
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return loss, psnr


    def _quantize(self, inputs, quantization_step, mode="noise"):
        if mode == "noise":
            noise = (torch.rand(inputs.size(), device=inputs.device) - 0.5) * quantization_step
            return inputs + noise
        elif mode == "symbols":
            return RoundNoGradient.apply(inputs / quantization_step) * quantization_step
        
    def apply_mask(self):
        soft_mask = torch.sigmoid(self.m)
        binary_mask = (soft_mask > self.epsilon).float()

        # Apply the mask to each of the parameters
        self._xyz.data *= binary_mask.view(-1, 1)  # Applying mask to _xyz
        self._cholesky.data *= binary_mask.view(-1, 1)  # Applying mask to _cholesky
        self._opacity.data *= binary_mask.view(-1, 1)  # Applying mask to _opacity
        self._features_dc.data *= binary_mask.view(-1, 1)  # Applying mask to _features_dc
