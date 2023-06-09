#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#hypersolver_augmentedNODE.py

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
from sciml_bench.benchmarks.neuralODEs.hypersolver_augmentedNODE.torchdyn.numerics import odeint
from sciml_bench.benchmarks.neuralODEs.hypersolver_augmentedNODE.torchdyn.numerics import HyperEuler, Euler


# implementation from other files
from sciml_bench.benchmarks.neuralODEs.hypersolver_augmentedNODE.nodes_misc import *

class VanillaHyperNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        for p in self.net.parameters():
            torch.nn.init.uniform_(p, 0, 1e-5)
    def forward(self, t, x):
        return self.net(x)
    
class ODENet(nn.Module):
    def __init__(self):
        super().__init__()
  
        self.linear3=nn.Linear(64, 3)
        self.linear2=nn.Linear(64, 64)
        self.linear1=nn.Linear(6, 64)
        self.apply(initialize_weights)
    def forward(self, t_span, x):
        try:
            x=torch.cat((x, x[:,:,0]*x[:,:,1], x[:,:,1]*x[:,:,2], x[:,:,0]*x[:,:,2]), axis=2)
        except:
            x=torch.cat((x, torch.unsqueeze(x[:,0]*x[:,1],1), torch.unsqueeze(x[:,1]*x[:,2],1), torch.unsqueeze(x[:,0]*x[:,2],1)), axis=1)

        x=self.linear1(x)
        x=nn.Softplus()(x)
        x=self.linear2(x)
        x=nn.Softplus()(x)
        x=self.linear3(x)
        return x
    




def create_model_node():
    net = ODENet()
    return net
    



def train(hypersolver, base_solver, odenet, train_loader,  hypernetwork_optimizer, ode_optimizer, sys, num_iterations, viz, save_file, t_span, full_x0, device ):
    odenet.train()
    hypersolver.train()
    train_losses  = []
    loss_func = nn.MSELoss()
    hypernet = hypersolver.hypernet
    for i, data in enumerate(train_loader):
        x0, t, y = data
 
        X = y[:-1].reshape(-1, 3)
        X_next_gt = y[1:].reshape(-1, 3)
        # step forward (fixed-step, time-invariant system hence any `t` as first argument is fine) with base solver
        dt = t_span[1] - t_span[0]
        _, X_next, _ = base_solver.step(sys, X, 0., dt) # step returns a Tuple (k1, berr, sol). The first two values are used internally
        # within `odeint`
        residuals = (X_next_gt - X_next) / dt**2
        residuals_hypersolver = hypernet(0., X)
        loss = loss_func(residuals, residuals_hypersolver)
        print('Hypersolver loss= '+str(loss))
        loss.backward(); hypernetwork_optimizer.step(); hypernetwork_optimizer.zero_grad()
        
        y_hat = torch.stack([odeint(odenet,torch.unsqueeze(x0[i, :],0), t[i,:], solver=hypersolver)[1] for i in range(x0.shape[0])], dim=0)
       # y_hat = torch.stack([odeint(self.fun,x0.to(device), t_span[i,:], method='rk4') for i in range(batch_size)], dim=0)
       
        loss=torch.mean(torch.abs(y - y_hat[:,:,0,:]))

        loss.backward(); ode_optimizer.step(); ode_optimizer.zero_grad()
        train_losses.append(loss.item())
        print('Total loss= '+str(loss))
        if viz:
            if i%50==0:
                with torch.no_grad():
                    visualize(odenet, hypersolver, sys, i,  save_file, t_span, full_x0, device)
                
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
    log.begin('Running benchmark hypersolver_augmentedNODE on training mode')

    # top-level process
    console = params_out.log.console
    console.begin('Running benchmark hypersolver_augmentedNODE on training mode')

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
    with log.subproc('Creating NN models'):
        odenet = create_model_node().to(device)
        net = nn.Sequential(nn.Linear(3, 64), nn.Softplus(), nn.Linear(64, 64), nn.Softplus(), nn.Linear(64, 3)).to(device)
        hypersolver= HyperEuler(VanillaHyperNet(net)).to(device)

    # data
    console.begin('Creating data loader')
    dataset_dir = params_in.dataset_dir
    train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
    console.ended('Creating data loader')

    # Training 
    log.begin('Training NN model')
    hypernetwork_optimizer = torch.optim.Adadelta(hypersolver.parameters(), lr=3e-4)
    ode_optimizer=torch.optim.Adam(odenet.parameters(), lr=0.001)
    base_solver = Euler()
    with log.subproc('Running model.fit()'):
        params_out.system.stamp_event('start train()')
        history = train(hypersolver, base_solver, odenet, train_dataloader,  hypernetwork_optimizer, ode_optimizer, sys, num_iterations, viz, params_in.output_dir, train_set.t_span, x0, device )

    # save model
    with log.subproc('Saving the model'):
        model_file = params_in.output_dir / 'hypersolver_augmentedNODE_model.pt'
        torch.save({'odenet':odenet.state_dict(), 'hypersolver':hypersolver.state_dict()}, model_file)
        log.message(f'Saved to: {model_file}')

    # save history
    with log.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history, handle)
        log.message(f'Saved to: {history_file}')
    log.ended('Training NN model')

    # top-level process
    console.ended('Running benchmark hypersolver_augmentedNODE on training mode')


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





 
    log.begin('Running benchmark hypersolver_augmentedNODE on inference mode')
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
        odenet = create_model_node().to(device)
        net = nn.Sequential(nn.Linear(3, 64), nn.Softplus(), nn.Linear(64, 64), nn.Softplus(), nn.Linear(64, 3)).to(device)
        hypersolver= HyperEuler(VanillaHyperNet(net)).to(device)
    # Load the model and perform bulk inference 
    with log.subproc('Model loading'):
        path=params_in.model
        checkpoint=torch.load(path)
        odenet.load_state_dict(checkpoint['odenet'])
        hypersolver.load_state_dict(checkpoint['hypersolver'])
  
        testloader = DataLoader(test_set, batch_size=12,
                        shuffle=True)
        x0, t, y = next(iter(testloader))
        print('Elephant', x0.shape, t.shape, y.shape)
    with log.subproc('Model inference'): 
        with torch.no_grad():   
            y_hat = torch.stack([odeint(odenet,torch.unsqueeze(x0[i, :],0), t[i,:], solver=hypersolver)[1] for i in range(x0.shape[0])], dim=0)
       
    with log.subproc('Calculate outputs'):    
        loss = torch.mean(torch.abs(y - y_hat[:,:,0,:]))
        
        if viz:
            with torch.no_grad():
                visualize(odenet, hypersolver, sys, -1,  params_in.output_dir , test_set.t_span, initial_x, device)
                
        print('Test loss= '+str(loss))
   

    log.ended('Running benchmark hypersolver_augmentedNODE on inference mode')
