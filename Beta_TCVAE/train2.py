import torch
from torch.autograd import Variable

import os
import time
import Beta_TCVAE.distribution as dist
from Beta_TCVAE.model2 import Beta_TCVAE
from utils import save_checkpoint, load_checkpoint

def train_beta_TCVAE(args, data_loader, device):

    # setup the model
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = dist.FactorialNormalizingFlow(dim=args.z_dim, nsteps=32)
        q_dist = dist.Normal()
        
    model = Beta_TCVAE(args, prior_dist=prior_dist, q_dist=q_dist)
    model.to(device)
    
    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.ckpt_iter is not None and os.path.exists(args.ckpt_dir):
        load_checkpoint(args.ckpt_dir, model, optimizer, args.method, args.ckpt_iter)
    model.train()
    
    # training loop
    dataset_size = len(data_loader.dataset)
    iteration = 0
    max_iter = len(data_loader) * args.num_epochs

    pbar = tqdm(total=max_iter)
    pbar.update(iteration)
    for epoch in range(args.num_epochs):
        for (data,labels) in data_loader:
            iteration += 1
            batch_time = time.time()
            optimizer.zero_grad()
            
            # transfer to GPU
            x = Variable(data.to(device))
            
            # do ELBO gradient and accumulate loss
            obj, elbo = model.elbo(x, dataset_size)
            obj.mean().mul(-1).backward()
            optimizer.step()
            
            # report training diagnostics
            if iteration % args.display_step == 0:
                print('[%03d] time: %.2f | beta %.2f | ELBO: %.4f' % (iteration, time.time() - batch_time, model.beta, obj.mean().cpu().item()))

            if iteration % args.save_step == 0:
                save_checkpoint(args.ckpt_dir, 'ckpt_'+ args.method + '_' +str(iteration), 
                                 model, optimizer, iteration)
    print("[Training Finished]")
    print()
    return model   
