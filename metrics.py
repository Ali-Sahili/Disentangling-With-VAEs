import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

import os
import numpy as np
from matplotlib import pyplot as plt

from WAE.model import WAE_MMD
from DIP_VAE.model import DIP_VAE
from Beta_TCVAE.model import Beta_TCVAE
from Factor_VAE.model import Factor_VAE
from Beta_VAE.model import Beta_VAE, Annealed_VAE

from generate_dataset import SimpleDots
from utils import load_checkpoint, save_checkpoint


#--------------------------------------------------------------------------
#                             Latent Traversal:
#                Qualitative metric to Evaluate disentanglement
#--------------------------------------------------------------------------

def plot_traversal(method='Beta_VAE', ckpt_dir="checkpoints", index=950, ckpt_iter=38000, 
                   z_dim=4, nc=1, output_dir = "outputs",
                   figure_width=10.5, num_cols=9, image_height=1.5):
    """
    Plot a traversal of the latent space.
    Steps are:
        1) encode an input to a latent representation
        2) adjust each latent value from -3 to 3 while keeping other values fixed
        3) decode each adjusted latent representation
        4) display
    """
    dataset = SimpleDots(image_size=(64, 64), circle_radius=8, num_images=2000)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    
    if method == 'Beta_VAE':
        model = Beta_VAE(z_dim, nc)
    elif method == 'Annealed_VAE':
        model = Annealed_VAE(z_dim, nc)
    elif method == 'Factor_VAE':
        model = Factor_VAE(nc, z_dim)
    elif method == 'Beta_TCVAE':
        model = Beta_TCVAE(nc, z_dim)
    elif method == 'DIP_VAE':
        model = DIP_VAE(nc, z_dim)
    elif method == 'WAE':
        model = WAE_MMD(nc, z_dim)
    else:
        raise NotImplementedError('only support the following models Beta-VAE, Annealed-VAE,\
                                   Factor_VAE, Beta_TCVAE, DIP_VAE and WAE.')
        
    model = model.cuda()
    load_checkpoint(ckpt_dir, model, None, method, ckpt_iter)
    model.eval()
    
    sample, target = data_loader.dataset[index]
    sample_batch = sample[None].cuda()

    if method == "DIP_VAE" or method == "Beta_TCVAE" or method == "Factor_VAE":
        mu, log_var = model.encode(sample_batch)
        zs = model.reparameterize(mu, log_var)
    elif method == "WAE":
        zs = model.encode(sample_batch)
    elif method == 'Beta_VAE' or method == 'Annealed_VAE':
        zs = model._encode(sample_batch)

    z = zs[:,:z_dim]  # since we're not training, no noise is added
    #print(z.shape)
    
    num_rows = z.shape[-1]
    num_cols = num_cols
    
    fig = plt.figure(figsize=(figure_width, image_height * num_rows))

    for i in range(num_rows):
        z_i_values = np.linspace(-3.0, 3.0, num_cols)
        z_i = z[0][i].detach().cpu().numpy()
        z_diffs = np.abs((z_i_values - z_i))
        j_min = np.argmin(z_diffs)
        for j in range(num_cols):
            z_i_value = z_i_values[j]
            if j != j_min:
                z[0][i] = z_i_value
            else:
                z[0][i] = float(z_i)
             
            if method == 'Beta_VAE' or method == 'Annealed_VAE':    
                x = F.sigmoid(model._decode(z.view(-1,z_dim))).detach().cpu().numpy()
            elif method=="DIP_VAE" or method=="Beta_TCVAE" or method== "Factor_VAE" or method=="WAE":
                x = model.decode(z.view(-1,z_dim)).detach().cpu().numpy()
                
            ax = fig.add_subplot(num_rows, num_cols, i * num_cols + j + 1)
            ax.imshow(x[0][0], cmap='gray')
            
            if i == 0:# or j == j_min:
                ax.set_title(f'{z[0][i]:.1f}')
            if j == j_min:
                ax.set_title('\n')
                    
            if j == j_min:
                ax.set_xticks([], [])
                ax.set_yticks([], []) 
                
                color = 'mediumseagreen'
                width = 4
                for side in ['top', 'bottom', 'left', 'right']:
                    ax.spines[side].set_color(color)
                    ax.spines[side].set_linewidth(width)
            else:
                ax.axis('off')
          
        z[0][i] = float(z_i)
        
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.04)
    
    save_plot_path = os.path.join(output_dir, "Latent_traversal")
    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)  
    plt.savefig(save_plot_path + '/' + method + '.png')
    plt.show()

    return zs[:,:z_dim], target.view(1,2)

#--------------------------------------------------------------------------
#                Metric introduced in Kim and Mnih (2018)
#--------------------------------------------------------------------------

def compute_disentanglement(zs, ys, L=1000, M=20000):
    
    N, D = zs.size()
    _, K = ys.size()
    zs_std = torch.std(zs, dim=0)
    ys_uniq = [c.unique() for c in ys.split(1, dim=1)]  # global: move out
    V = torch.zeros(D, K, device=zs_std.device)
    ks = np.random.randint(0, K, M)      # sample fixed-factor idxs ahead of time

    for m in range(M):
        k = ks[m]
        fk_vals = ys_uniq[k]
        # fix fk
        fk = fk_vals[np.random.choice(len(fk_vals))]
        # choose L random zs that have this fk at factor k
        zsh = zs[ys[:, k] == fk]
        zsh = zsh[torch.randperm(zsh.size(0))][:L]
        d_star = torch.argmin(torch.var(zsh / zs_std, dim=0))
        V[d_star, k] += 1

    return torch.max(V, dim=1)[0].sum() / M

#--------------------------------------------------------------------------
#                                END
#--------------------------------------------------------------------------
if __name__ == "__main__":
    zs, ys = plot_traversal(method="Factor_VAE")
    mig = compute_disentanglement(zs, ys)
    print(mig)
