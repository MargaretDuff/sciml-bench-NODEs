#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# impl_mnist_torch.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
"""

from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
import os 

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class LorenzDataset(Dataset):
    def __init__(self, sys, x0, data_size, batch_time,device, spread):
        self.t_span = torch.linspace(0., 3., data_size)
        self.x0=spread*torch.randn(512, 3) + torch.tensor([x0])
        self.data_size=data_size
        self.batch_time=batch_time
        with torch.no_grad():
            self.sol_gt = odeint(sys, self.x0, self.t_span, method='dopri5', atol=1e-3, rtol=1e-3)

        self.device=device
    def __len__(self):
        return (self.data_size - self.batch_time)*512
    def __getitem__(self, idx):
        batch_x0 = self.sol_gt[ idx//512, idx%512] 
        batch_t = self.t_span[idx//512:idx//512+self.batch_time] 
        batch_y=torch.stack([self.sol_gt[idx//512+i, idx%512]  for i in range(self.batch_time)], dim=0)
        
        
        return(batch_x0.to(self.device), batch_t.to(self.device), batch_y.to(self.device))




class Lorenz(nn.Module):
    def __init__(self, sigma, rho, beta):
        super().__init__()
        self.p = nn.Linear(1,1)
        self.sigma=sigma
        self.rho=rho
        self.beta=beta
    def forward(self, t, x):
        x1, x2, x3 = x[...,:1], x[...,1:2], x[...,2:]
        dx1 = self.sigma * (x2 - x1)
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta* x3
        return torch.cat([dx1, dx2, dx3], -1)

def visualize(model, sys,  itr, save_folder, t_span, true_x0, device):
        true_x0 = torch.tensor([true_x0]).to(device)
        t_span = t_span.to(device)
        with torch.no_grad():
            true_y = odeint(sys.to(device), true_x0 , t_span, method='dopri5', atol=1e-3, rtol=1e-3)

        with torch.no_grad():
            pred_y =odeint( model,true_x0, t_span, method='rk4')
            
            x0_extra=torch.randn(24, 3).to(device) + true_x0

            with torch.no_grad():
                pred_y_extra = odeint(model, x0_extra, t_span, method='rk4')

        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=True)
        ax_phase = fig.add_subplot(132, frameon=True)
        ax_vecfield = fig.add_subplot(133, frameon=True)
        
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t_span')
        ax_traj.set_ylabel('x,y,z')
        ax_traj.plot(t_span.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], 'r-', t_span.cpu().numpy(),true_y.cpu().numpy()[:, 0, 1],'b-', t_span.cpu().numpy(), true_y.cpu().numpy()[:, 0, 2], 'g-')
        #print(pred_y.shape)
        ax_traj.plot(t_span.cpu().numpy(), pred_y.cpu().detach().numpy()[:, 0, 0], 'r--', t_span.cpu().numpy(), pred_y.cpu().detach().numpy()[:, 0, 1], 'b--',t_span.cpu().numpy(), pred_y.cpu().detach().numpy()[:, 0, 2], 'g--')
        ax_traj.set_xlim(t_span.cpu().min(), t_span.cpu().max())
        #ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Butterfly plot')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('z')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 2], 'g-')
        ax_phase.plot(pred_y.cpu().detach().numpy()[:, 0, 0], pred_y.cpu().detach().numpy()[:, 0, 2], 'b-')
        for l in range(24):
            ax_phase.plot(pred_y_extra.cpu().detach().numpy()[:, l, 0], pred_y_extra.cpu().detach().numpy()[:, l, 2], 'b--', alpha=0.1)
       # ax_phase.set_xlim(-2, 2)
       # ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('z')

        z,x = np.mgrid[-20:20:21j, -20:20:21j]
   #     print(x.shape)
    #    print(z.shape)
        dydt = model(0, torch.Tensor(np.stack([x, true_y.cpu().numpy()[-1, 0, 2]*np.ones(x.shape), z], -1).reshape(21 * 21, 3)).to(device) ).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 2]**2+ dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 3)
        ax_vecfield.streamplot(x, z, dydt[:, :, 0], dydt[:, :, 2], color="blue")
        dydt = sys(0, torch.Tensor(np.stack([x, true_y.cpu().numpy()[-1, 0, 2]*np.ones(x.shape), z], -1).reshape(21 * 21, 3)).to(device) ).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 2]**2+ dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 3)
        ax_vecfield.streamplot(x, z, dydt[:, :, 0], dydt[:, :, 2], color="black")

        ax_vecfield.set_xlim(-20, 20)
        ax_vecfield.set_ylim(-20, 20)

        fig.tight_layout()

        plt.savefig(save_folder / 'training_visualisation_iter_{}.png'.format(itr))
        plt.close()
        #plt.show()





