import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from Factor_VAE.model2 import FactorVAE, Discriminator

from utils import load_checkpoint_TC, save_checkpoint_TC
from utils import kl_divergence, recons_loss, permute_dims

def train_TC(args, data_loader_1, data_loader_2):

    device = 'cuda' if args.use_cuda else 'cpu'
    

    model = FactorVAE(args.z_dim, args.nc).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    model.train()
        
    D = Discriminator(args.z_dim).to(device)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr_D, 
                                   betas=(args.beta1_D, args.beta2_D))
    D.train()

    if args.ckpt_iter is not None and os.path.exists(args.ckpt_dir):
        load_checkpoint_TC(args.ckpt_dir,model,D,optimizer,optimizer_D,args.method,args.ckpt_iter)
    
    ones = torch.ones(args.batch_size, dtype=torch.long, device=device)
    zeros = torch.zeros(args.batch_size, dtype=torch.long, device=device)

    global_iter = 0
    max_iter = len(data_loader) * args.num_epochs

    pbar = tqdm(total=max_iter)
    pbar.update(global_iter)
    for epoch in range(args.num_epochs):   
        for ((x_true1, labels1), (x_true2, labels2)) in zip(data_loader_1,data_loader_2):
            global_iter += 1
            pbar.update(1)

            x_true1 = x_true1.to(device)
            x_recon, mu, logvar, z = model(x_true1)
            recon_loss = recons_loss(x_true1, x_recon)
            kld = kl_divergence(mu, logvar)

            D_z = D(z)
            TC_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

            vae_loss = recon_loss + kld + args.gamma * TC_loss

            optimizer.zero_grad()
            vae_loss.backward(retain_graph=True)
            optimizer.step()
            
            ##
            x_true2 = x_true2.to(device)
            z_prime = model(x_true2, no_dec=True)
            z_pperm = permute_dims(z_prime).detach()
            D_z_pperm = D(z_pperm)

            D_TC_loss = 0.5*(F.cross_entropy(D_z.detach(), zeros) + F.cross_entropy(D_z_pperm, ones))
            
            optimizer_D.zero_grad()
            D_TC_loss.backward()
            optimizer_D.step()

            if global_iter % args.display_step == 0:
                pbar.write('[{}] recon_loss:{:.3f} kld:{:.3f} VAE_TC_loss:{:.3f} D_TC_loss:{:.3f}'.format(global_iter, recon_loss.item(), kld.item(), TC_loss.item(), D_TC_loss.item()))

            if global_iter % args.save_step == 0:
                save_checkpoint_TC(args.ckpt_dir, 'ckpt_'+ args.method + '_' +str(global_iter), 
                                     model, D, optimizer, optimizer_D, global_iter)
                pbar.write('Saved checkpoint(iter:{})'.format(global_iter))


    pbar.write("[Training Finished]")
    pbar.close()
        
    return model

