import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import load_checkpoint, save_checkpoint

from Factor_VAE.model import Factor_VAE, D_model


def permute_latent(z):
    """
    Permutes each of the latent codes in the batch
    :param z: [B x D]
    :return: [B x D]
    """
    B, D = z.size()

    # Returns a shuffled inds for each latent code in the batch
    inds = torch.cat([(D *i) + torch.randperm(D) for i in range(B)])
    return z.view(-1)[inds].view(B, D)


def train_Factor_VAE(args, data_loader, device):

    model = Factor_VAE(args.nc, args.z_dim).to(device)
    Discriminator = D_model(args.z_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(Discriminator.parameters(), lr=args.lr_D, 
                              betas = (args.beta1_D, args.beta2_D))
    
    if args.ckpt_iter is not None and os.path.exists(args.ckpt_dir):
        load_checkpoint(args.ckpt_dir, model, optimizer, args.method, args.ckpt_iter)

    model.train()
    Discriminator.train()
    
    true_labels = torch.ones(args.batch_size, dtype= torch.long, requires_grad=False).to(device)
    false_labels = torch.zeros(args.batch_size,dtype=torch.long, requires_grad=False).to(device)
    
    global_iter = 0
    max_iter = len(data_loader) * args.num_epochs

    pbar = tqdm(total=max_iter)
    pbar.update(global_iter)
    
    kl_loss_plot = []
    recon_loss_plot = []
    vae_loss_plot = []
    
    for epoch in range(args.num_epochs):
        kl_loss_t = 0
        recon_loss_t = 0
        vae_loss_t = 0
        for (data, label) in data_loader:
            global_iter += 1
            pbar.update(1)

            optimizer.zero_grad()
            optimizer_D.zero_grad()

            x = Variable(data.to(device))
            x_recon, mu, log_var, z = model(x)
            
            # Update the VAE
            recons_loss = F.mse_loss(x_recon, x)
            kld_loss = torch.mean(-0.5 *torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), 
                                                                                       dim=0)
            D_z_reserve = Discriminator(z)
            vae_tc_loss = (D_z_reserve[:, 0] - D_z_reserve[:, 1]).mean()

            anneal_reg = min(global_iter/args.f_anneal_steps, 1.)
            loss = recons_loss + kld_loss + anneal_reg * args.fvae_gamma * vae_tc_loss
            
            kl_loss_t += kld_loss.item()
            recon_loss_t += recons_loss.item()
            vae_loss_t += loss.item()
            
            loss.backward()
            optimizer.step()

            # Update the Discriminator
            for _ in range(1):
                z = z.detach() # Detach so that VAE is not trained again
                z_perm = permute_latent(z)
                D_z_perm = Discriminator(z_perm)
                D_tc_loss = 0.5 * (F.cross_entropy(D_z_reserve.detach(), false_labels) +
                                   F.cross_entropy(D_z_perm, true_labels))

                D_tc_loss.backward()
                optimizer_D.step()

            if global_iter % args.display_step == 0:
                pbar.write('[{}] recon_loss:{:.3f} | kld:{:.3f} | TC:{:.3f} | D:{:.3f}'.format(
                            global_iter, recons_loss.item(), kld_loss.item(), 
                            vae_tc_loss.item(), D_tc_loss.item()))

            if global_iter % args.save_step == 0:
                save_checkpoint(args.ckpt_dir, 'ckpt_'+ args.method + '_' +str(global_iter), 
                                 model, optimizer, global_iter)

        kl_loss_plot.append(kl_loss_t/len(data_loader))
        recon_loss_plot.append(recon_loss_t/len(data_loader))
        vae_loss_plot.append(vae_loss_t/len(data_loader))

    pbar.write("[Training Finished]")
    pbar.write("")
    pbar.close()
    print()
    
    return model, {"Recons":recon_loss_plot, "KLD":kl_loss_plot, "total_loss":vae_loss_plot}
  
