import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import load_checkpoint, save_checkpoint
from utils import kl_divergence, recons_loss

from Beta_VAE.model import Beta_VAE, Annealed_VAE


def train(args, data_loader, device):

    if args.dataset == 'toy_dataset': # "celebA"
            #assert args.nc == 3
            assert args.nc == 1
            decoder_dist = 'gaussian'
    else:
            raise NotImplementedError

    if args.method == 'Beta_VAE':
        model = Beta_VAE
    elif args.method == 'Annealed_VAE':
        model = Annealed_VAE
    else:
        raise NotImplementedError('only support two models Beta-VAE or Annealed-VAE')
        
    model = model(args.z_dim, args.nc).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    if args.ckpt_iter is not None and os.path.exists(args.ckpt_dir):
        load_checkpoint(args.ckpt_dir, model, optimizer, args.method, args.ckpt_iter)

    model.train()
    
    C_max = Variable(torch.FloatTensor([args.C_max]).to(device))


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
            
            #print(data.shape)
            #print(label.shape)

            x = Variable(data.to(device))
            x_recon, mu, logvar = model(x)
            recon_loss = recons_loss(x, x_recon, decoder_dist)
            kld = kl_divergence(mu, logvar)

            if args.method == 'Beta_VAE':
                beta_vae_loss = recon_loss + args.beta*kld
            elif args.method == 'Annealed_VAE':
                C = torch.clamp(C_max/args.C_stop_iter*global_iter, 0, C_max.data[0])
                beta_vae_loss = recon_loss + args.gamma*(kld-C).abs()

            kl_loss_t += kld.item()
            recon_loss_t += recon_loss.item()
            vae_loss_t += beta_vae_loss.item()

            optimizer.zero_grad()
            beta_vae_loss.backward()
            optimizer.step()

            if global_iter % args.display_step == 0:
                pbar.write('[{}] recon_loss:{:.3f} kld:{:.3f}'.format(
                            global_iter, recon_loss.item(), kld.item()))

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
  
