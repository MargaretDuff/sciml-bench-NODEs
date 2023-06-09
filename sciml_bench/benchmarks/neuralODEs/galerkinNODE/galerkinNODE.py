#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#galerkinNODE.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Benchmark: vanilla neural ODE
This is a single device training/inference example.
"""

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

# libs required by implementation

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint as odeint

# implementation from other files
from sciml_bench.benchmarks.neuralODEs.galerkinNODE.nodes_misc import *

from sciml_bench.benchmarks.neuralODEs.galerkinNODE.torchdyn.nn import    GalLinear, Fourier
class ODENet(nn.Module):
    def __init__(self):
        super().__init__()

        self.galerkin=GalLinear(6, 64, expfunc=Fourier(4))
        self.linear3=nn.Linear(64, 3)
        self.linear2=nn.Linear(64, 64)
  
        self.apply(initialize_weights)
    def forward(self, t, x):
        try:
            x=torch.cat((x, torch.unsqueeze(x[:,:,0]*x[:,:,1],2), torch.unsqueeze(x[:,:,1]*x[:,:,2],2), torch.unsqueeze(x[:,:,0]*x[:,:,2],2)), axis=2)
        except:
            x=torch.cat((x, torch.unsqueeze(x[:,0]*x[:,1],1), torch.unsqueeze(x[:,1]*x[:,2],1), torch.unsqueeze(x[:,0]*x[:,2],1)), axis=1)
        t_shape = list(x.shape)
        t_shape[1] = 1
        t = t * torch.ones(t_shape).to(x)
        x=torch.cat([x, t], 1).to(x)
        x=self.galerkin(x)
        x=nn.Softplus()(x)
        x=self.linear2(x)
        x=nn.Softplus()(x)
        x=self.linear3(x)

        return x
    
def create_model_node():
    net = ODENet()
    return net
    

def train(model, train_loader, optimizer, sys,num_iterations, viz, save_file, t_span, full_x0,device ):
    model.train()
    train_losses  = []
    for i, data in enumerate(train_loader):
        x0, t, y = data
        optimizer.zero_grad()
        y_hat = torch.stack([odeint(model,torch.unsqueeze(x0[i, :],0), t[i,:], method='rk4') for i in range(x0.shape[0])], dim=0)
        loss = torch.mean(torch.abs(y - y_hat[:,:,0,:]))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if viz:
            if i%50==0:
                with torch.no_grad():
                    visualize(model, sys, i,  save_file, t_span, full_x0, device)
        print('Train loss= '+str(loss))
        if i>num_iterations:
            return train_losses
    return train_losses


def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance
    in the training mode. Please consult the API documentation.
    """
    params_out.activate(rank=0, local_rank=0,
                      activate_log_on_host=False,
                      activate_log_on_device=True, console_on_screen=True)

    log = params_out.log
    log.begin('Running benchmark galerkinNODE on training mode')

    # top-level process
    console = params_out.log.console
    console.begin('Running benchmark galerkinNODE on training mode')

    # We expect two benchmark-specific arguments here: 
    # batch_size and epochs. If not, we will assign 
    # default values.
    with log.subproc('Parsing input arguments'):
        # hyperparameters
        suggested_args = {
            'batch_size': 32,
            'num_iterations': 5,
            'viz': True ,
            'data_size':10000,
            'batch_time':20,
            'train_data_spread':20,
            'x0':[15., 15., 15.],
            'sigma': 10.,
            'rho': 10.,
            'beta': 8/3, 
            'use_gpu':True
        } 
        args = params_in.bench_args.try_get_dict(default_args=suggested_args)
        batch_size = args['batch_size']
        num_iterations = args['num_iterations']
        viz=args['viz']
        data_size=args['data_size']
        batch_time=args['batch_time']
        train_data_spread=args['train_data_spread']
        x0=args['x0']
        sigma=args['sigma']
        rho=args['rho']
        beta=args['beta']
        use_gpu=args['use_gpu']
        if args['use_gpu'] and torch.cuda.is_available():
            device = "cuda:0"
            log.message('Using GPU')
        else:
            device = "cpu"
            log.message('Using CPU')

        log.message(f'batch_size = {batch_size}')
        log.message(f'num_iterations     = {num_iterations}')
        log.message(f'viz     = {viz}')
        log.message(f'data_size     = {data_size}')
        log.message(f'batch_time     = {batch_time}')
        log.message(f'train_data_spread     = {train_data_spread}')
        log.message(f'x0     = {x0}')
        log.message(f'sigma     = {sigma}')
        log.message(f'rho     = {rho}')
        log.message(f'beta     = {beta}')
        log.message(f'use_gpu     = {use_gpu}')
  


    with log.subproc('Writing the argument file'):
        args_file = params_in.output_dir / 'arguments_used.yml'
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # create datasets
    with log.subproc('Creating datasets'):
        sys = Lorenz(sigma,rho,beta)
        dataset_dir = params_in.dataset_dir
        train_set = LorenzDataset(sys,x0,  data_size, batch_time, device,  train_data_spread)
     
        log.message(f'Dataset directory: {dataset_dir}')

    # create model
    with log.subproc('Creating NN model'):
        net = create_model_node().to(device)

    # data
    console.begin('Creating data loader')
    dataset_dir = params_in.dataset_dir
    train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
    console.ended('Creating data loader')

    # Training 
    log.begin('Training NN model')
    optimizer= torch.optim.Adam(net.parameters(), lr=0.001)
    with log.subproc('Running model.fit()'):
        params_out.system.stamp_event('start train()')
        history = train(net, train_dataloader, optimizer, sys, num_iterations, viz, params_in.output_dir, train_set.t_span, x0, device )

    # save model
    with log.subproc('Saving the model'):
        model_file = params_in.output_dir / 'vanillaNODE_model.pt'
        torch.save(net.state_dict(), model_file)
        log.message(f'Saved to: {model_file}')

    # save history
    with log.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history, handle)
        log.message(f'Saved to: {history_file}')
    log.ended('Training NN model')

    # top-level process
    console.ended('Running benchmark galerkinNODE on training mode')


def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    suggested_args = {
        'batch_size': 80,
        'viz': True ,
        'data_size':10000,
        'test_data_spread':20,
        'initial_x':[15., 15., 15.],
        'sigma': 10.,
        'rho': 10.,
        'beta': 8/3, 
        'use_gpu':True
    } 
    args = params_in.bench_args.try_get_dict(default_args=suggested_args)


    viz=args['viz']
    data_size=args['data_size']
    test_data_spread=args['test_data_spread']
    initial_x=args['initial_x']
    sigma=args['sigma']
    rho=args['rho']
    beta=args['beta']
    use_gpu=args['use_gpu']

    params_out.activate(rank=0, local_rank=0, activate_log_on_host=True,
                      activate_log_on_device=True, console_on_screen=True)

    log = params_out.log

    if args['use_gpu'] and torch.cuda.is_available():
        device = "cuda:0"
        log.message('Using GPU')
    else:
        device = "cpu"
        log.message('Using CPU')





 
    log.begin('Running benchmark galerkinNODE on inference mode')
    log.message(f'viz     = {viz}')
    log.message(f'data_size     = {data_size}')
    log.message(f'test_data_spread     = {test_data_spread}')
    log.message(f'initial_x     = {initial_x}')
    log.message(f'sigma     = {sigma}')
    log.message(f'rho     = {rho}')
    log.message(f'beta     = {beta}')
    log.message(f'use_gpu     = {use_gpu}')

    
    # create datasets
    with log.subproc('Creating datasets'):
        sys = Lorenz(sigma,rho,beta)
        test_set = LorenzDataset(sys,initial_x,  data_size, data_size-1, device,  test_data_spread)
        

    # create model
    with log.subproc('Creating NN model'):
        net = create_model_node().to(device)
    # Load the model and perform bulk inference 
    with log.subproc('Model loading'):
        path=params_in.model
        net.load_state_dict(torch.load(path))
  
        testloader = DataLoader(test_set, batch_size=24,
                        shuffle=True)
        x0, t, y = next(iter(testloader))
    with log.subproc('Model inference'): 
        with torch.no_grad():      
            y_hat = torch.stack([odeint(net,torch.unsqueeze(x0[i, :],0), t[i,:], method='rk4') for i in range(x0.shape[0])], dim=0)
    with log.subproc('Calculate outputs'):    
        loss = torch.mean(torch.abs(y - y_hat[:,:,0,:]))
        
        if viz:
            with torch.no_grad():
                visualize(net, sys, -1,  params_in.output_dir, test_set.t_span, initial_x, device)
        print('Test loss= '+str(loss))
   

    log.ended('Running benchmark galerkinNODE on inference mode')
