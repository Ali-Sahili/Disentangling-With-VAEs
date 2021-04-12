import math
import torch
from torch import nn
from torch.nn import functional as F


class Beta_TCVAE(nn.Module):
    def __init__(self, nc=1, z_dim=10, anneal_steps=200, alpha=1., beta=6., gamma=1.):
        super(Beta_TCVAE, self).__init__()

        self.nc = nc
        self.z_dim = z_dim
        self.anneal_steps = anneal_steps

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        modules = []
        hidden_dims = [32, 32, 32, 32]

        # Build Encoder
        in_channels = nc
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 4, stride= 2, padding  = 1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc = nn.Linear(hidden_dims[-1]*16, 256)
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_var = nn.Linear(256, z_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(z_dim, 256 *  2)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= nc,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.fc(result)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):

        result = self.decoder_input(z)
        result = result.view(-1, 32, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var, z]

    def log_density_gaussian(self, x, mu, logvar):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    # Computes the VAE loss function
    def loss_function(self, input, recons, mu, log_var, z,dataset_size, global_iter, train=True):
        """
        KL(N(\mu,\sigma),N(0,1)) = \log\frac{1}{\sigma} + \frac{\sigma^2 +\mu^2}{2} - \frac{1}{2}
        """

        weight = 1 # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input, reduction='sum')

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim = 1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim = 1)

        batch_size, z_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, z_dim),
                                                mu.view(1, batch_size, z_dim),
                                                log_var.view(1, batch_size, z_dim))

        # Reference
        # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size -1)).to(input.device)
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss  = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        if train:
            global_iter += 1
            anneal_rate = min(0 + 1 * global_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss = recons_loss/batch_size + \
               self.alpha * mi_loss + \
               weight * (self.beta * tc_loss +
                         anneal_rate * self.gamma * kld_loss)
        
        return {'total_loss': loss,
                'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss,
                'TC_Loss':tc_loss,
                'MI_Loss':mi_loss}, global_iter
