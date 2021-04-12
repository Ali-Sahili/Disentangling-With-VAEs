import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import os
import imageio

#--------------------------------------------------------------------------
#                         weight initialization
#--------------------------------------------------------------------------
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean=0., std=1.):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

#--------------------------------------------------------------------------
#                         Computing losses
#--------------------------------------------------------------------------
def recons_loss(x, x_recon, distribution="bernoulli"):
    b_size = x.size(0)
    assert b_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(b_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(b_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()

    return kld


#--------------------------------------------------------------------------
#                    Loading and Saving checkpoints
#--------------------------------------------------------------------------
def save_checkpoint(ckpt_dir, filename, model, optimizer, global_iter):
    model_states = {'model':model.state_dict(),}
    optim_states = {'optimizer':optimizer.state_dict(),}
    states = {
                'iter':global_iter,
                'model_states':model_states,
                'optim_states':optim_states
             }

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    file_path = os.path.join(ckpt_dir, filename)
    with open(file_path, mode='wb+') as f:
        torch.save(states, f)
    print("=> saved checkpoint '{}' (iter {})".format(file_path, global_iter))

def load_checkpoint(ckpt_dir, model, optimizer, method, ckpt_iter=1000):
    file_path = os.path.join(ckpt_dir, 'ckpt_'+ method + '_' + str(ckpt_iter))
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        global_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['model_states']['model'])
        if optimizer is not None: 
            optimizer.load_state_dict(checkpoint['optim_states']['optimizer'])
        print("=> loaded checkpoint '{} (iter {})'".format(file_path, global_iter))
    else:
        print("=> no checkpoint found at '{}'".format(file_path))


def save_checkpoint_TC(ckpt_dir, filename, model, D, optimizer, optimizer_D, global_iter):
    model_states = {'D':D.state_dict(),
                    'VAE':model.state_dict()}
    optimizer_states = {'optimizer_D':optimizer_D.state_dict(),
                        'optimizer':optimizer.state_dict()}
    states = {'iter':global_iter,
              'model_states':model_states,
              'optimizer_states':optimizer_states}

    filepath = os.path.join(ckpt_dir, filename)
    with open(filepath, 'wb+') as f:
        torch.save(states, f)


def load_checkpoint_TC(ckpt_dir, model, D, optimizer, optimizer_D, method, ckpt_iter=1000):
    filepath = os.path.join(ckpt_dir, "ckpt_"+ method + "_" +str(ckpt_iter))
        
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            checkpoint = torch.load(f)

        global_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['model_states']['model'])
        D.load_state_dict(checkpoint['model_states']['D'])
        optimizer.load_state_dict(checkpoint['optimizer_states']['optimizer'])
        optimizer_D.load_state_dict(checkpoint['optimizer_states']['optimizer_D'])
        
        print("=> loaded checkpoint '{} (iter {})'".format(filepath, global_iter))
    else:
        print("=> no checkpoint found at '{}'".format(filepath))
#--------------------------------------------------------------------------
#                         Helper functions
#--------------------------------------------------------------------------
# Random permutation of B latent vectors
def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)
    
# Create Gif from generated images
def grid2gif(output_dir, gif_name, duration=0.2):
    imgs = [imageio.imread(output_dir+"/"+f) for f in os.listdir(output_dir)]
    imageio.mimsave(os.path.join(output_dir, gif_name), imgs, duration = duration)
         
#--------------------------------------------------------------------------
#                                END
#--------------------------------------------------------------------------
