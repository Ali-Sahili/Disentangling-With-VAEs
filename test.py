import os
import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import grid2gif

"""
def test_reconstruction(model):
    model.eval()
    gather = DataGather()
    
    x = gather.data['images'][0][:100]
    x = make_grid(x, normalize=True)
    x_recon = gather.data['images'][1][:100]
    x_recon = make_grid(x_recon, normalize=True)
    images = torch.stack([x, x_recon], dim=0).cpu()

    return images 
"""

def test(args, data_loader, model, device, global_iter):
        
    limit = 3
    inter = 2/3
    loc = -1
    
    model.eval()
        
    decoder = model.decoder
    encoder = model.encoder
    interpolation = torch.arange(-limit, limit+0.1, inter)

    n_dsets = len(data_loader.dataset)
    rand_idx = random.randint(1, n_dsets-1)

    random_img, labels = data_loader.dataset.__getitem__(rand_idx)
    random_img = Variable(random_img.to(device)).unsqueeze(0)
    if args.method == "DIP_VAE" or args.method == "Beta_TCVAE" or args.method == "Factor_VAE":
        mu, log_var = model.encode(random_img)
        random_img_z = model.reparameterize(mu, log_var)[:, :args.z_dim]
    elif args.method == "WAE":
        random_img_z = model.encode(random_img)[:, :args.z_dim]
    else:
        random_img_z = encoder(random_img)[:, :args.z_dim]

    random_z = Variable(torch.rand(1, args.z_dim).to(device))


    fixed_idx = 0
    fixed_img, labels = data_loader.dataset.__getitem__(fixed_idx)
    fixed_img = Variable(fixed_img.to(device)).unsqueeze(0)
    if args.method == "DIP_VAE" or args.method == "Beta_TCVAE" or args.method == "Factor_VAE":
        mu, log_var = model.encode(fixed_img)
        fixed_img_z = model.reparameterize(mu, log_var)[:, :args.z_dim]
    elif args.method == "WAE":
        fixed_img_z = model.encode(fixed_img)[:, :args.z_dim]
    else:
        fixed_img_z = encoder(fixed_img)[:, :args.z_dim]

    Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

    gifs = []
    for key in Z.keys():
        z_ori = Z[key]
        samples = []
        for row in range(args.z_dim):
            if loc != -1 and row != loc:
                continue
            z = z_ori.clone()
            for val in interpolation:
                z[:, row] = val
                if args.method=="DIP_VAE" or args.method=="Beta_TCVAE" or args.method== "Factor_VAE" or args.method=="WAE":
                    sample = model.decode(z.view(-1,args.z_dim))
                elif args.method=="Beta_VAE" or args.method=="Annealed_VAE":
                    sample = F.sigmoid(decoder(z.view(-1,args.z_dim))).data
                else:
                    sample = F.sigmoid(decoder(z.view(-1,args.z_dim,1,1))).data
                samples.append(sample)
                gifs.append(sample)
        samples = torch.cat(samples, dim=0).cpu()
        title = '{}_latent_traversal(iter:{})'.format(key, global_iter)

    if args.save_output:
        output_dir = os.path.join(args.output_dir, args.method + "_" +str(global_iter))
        os.makedirs(output_dir, exist_ok=True)
        gifs = torch.cat(gifs)
        gifs = gifs.view(len(Z), args.z_dim, len(interpolation), args.nc, 64, 64).transpose(1, 2)
        for i, key in enumerate(Z.keys()):
            for j, val in enumerate(interpolation):
                save_image( tensor=gifs[i][j].cpu(),
                            fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                            nrow=args.z_dim, pad_value=1)

            grid2gif(output_dir, "GIF_result_"+str(key)+".gif", duration=0.2)
    model.train()
