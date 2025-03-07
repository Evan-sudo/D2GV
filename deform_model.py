import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# 引入 DeformNetwork
from deformation import *  # 你的 DeformNetwork 定义应在这里
from utils import searchForMaxIteration
from utils import get_expon_lr_func

# default DW: 2, 64
class DeformModel:
    def __init__(self):
        # w = 156
        # self.deform = DeformNetwork(D=2, W=156, pos_multires=10, time_multires=6).cuda()
        self.deform = DeformNetwork(D=2, W=156, pos_multires=10, time_multires=6).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5 

    def step(self, xyz, timestamp):
        timestamp_expanded = torch.ones(xyz.size(0), 1, device=xyz.device) * timestamp
        return self.deform(xyz, timestamp_expanded)

    def train_setting(self):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': 0.00016 * self.spatial_lr_scale,
             "name": "deform"} 
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=0.00016 * self.spatial_lr_scale,
            lr_final=0.000016,
            lr_delay_mult=0.01,
            # max_steps=100000 20000
            max_steps=50000
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration

        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration) 
                param_group['lr'] = lr
                return lr

