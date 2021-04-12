import torch
from torch import nn
from torch.nn import functional as F

class DIP_VAE(nn.Module):

    def __init__(self, nc=1, z_dim=10, lambda_diag=10., lambda_offdiag=5.):
        super(DIP_VAE, self).__init__()

        self.nc = nc
        self.z_dim = z_dim
        self.lambda_diag = lambda_diag
        self.lambda_offdiag = lambda_offdiag

        
        hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        in_channels = nc
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, z_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, z_dim)


        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(z_dim, hidden_dims[-1] * 4)

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
                    nn.BatchNorm2d(hidden_dims[i + 1]),
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
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= nc,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):

        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
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
        return  self.decode(z), mu, log_var, z

    # Computes the DIP-VAE loss function
    def loss_function(self, input, recons_out, mu, log_var, mode="II"):
        """
        KL(N(\mu,\sigma),N(0,1)) = \log\frac{1}{\sigma} + \frac{\sigma^2 +\mu^2}{2} - \frac{1}{2}
        """

        kld_weight = 1 #* kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons_out, input, reduction='sum')

        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # DIP Loss
        centered_mu = mu - mu.mean(dim=1, keepdim = True) # [B x D]
        cov_mu = centered_mu.t().matmul(centered_mu).squeeze() # [D X D]

        # For DIp Loss I
        if mode== "I":
            cov_z = cov_mu
        # Add Variance for DIP Loss II
        elif mode == "II": 
            cov_z = cov_mu + torch.mean(torch.diagonal((2. * log_var).exp(), dim1 = 0), dim = 0)

        cov_diag = torch.diag(cov_z) # [D]
        cov_offdiag = cov_z - torch.diag(cov_diag) # [D x D]
        dip_loss = self.lambda_offdiag * torch.sum(cov_offdiag ** 2) + \
                   self.lambda_diag * torch.sum((cov_diag - 1) ** 2)

        loss = recons_loss + kld_weight * kld_loss + dip_loss
        return {'total_loss': loss,
                'Reconstruction_Loss':recons_loss,
                'KLD':-kld_loss,
                'DIP_Loss':dip_loss}

