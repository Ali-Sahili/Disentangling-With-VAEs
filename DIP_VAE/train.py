import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import load_checkpoint, save_checkpoint

from DIP_VAE.model import DIP_VAE


def train_DIP_VAE(args, data_loader, device):

    model = DIP_VAE(args.nc, args.z_dim, args.lambda_diag, args.lambda_offdiag).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    if args.ckpt_iter is not None and os.path.exists(args.ckpt_dir):
        load_checkpoint(args.ckpt_dir, model, optimizer, args.method, args.ckpt_iter)

    model.to(device)
    model.train()
    
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

            x = Variable(data.to(device))
            x_recon, mu, logvar, z = model(x)
            losses = model.loss_function(x, x_recon, mu, logvar, mode="I")

            kl_loss_t += losses["KLD"].item()
            recon_loss_t += losses["Reconstruction_Loss"].item()
            vae_loss_t += losses["total_loss"].item()

            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()

            if global_iter % args.display_step == 0:
                pbar.write('[{}] recon_loss:{:.3f} | kld:{:.3f} | DIP:{:.3f}'.format(
                            global_iter, losses["Reconstruction_Loss"].item(), 
                            losses["KLD"].item(), losses["DIP_Loss"].item()))

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
  
