import torch
from torch import nn
from torch.nn import functional as F

# Discriminator network for the Total Correlation (TC) loss
class D_model(nn.Module):
    def __init__(self, z_dim=10):
        super(D_model, self).__init__()
        
        self.z_dim = z_dim
        
        self.discriminator = nn.Sequential(nn.Linear(self.z_dim, 1000),
                                          nn.BatchNorm1d(1000),
                                          nn.LeakyReLU(0.2),
                                          nn.Linear(1000, 1000),
                                          nn.BatchNorm1d(1000),
                                          nn.LeakyReLU(0.2),
                                          nn.Linear(1000, 1000),
                                          nn.BatchNorm1d(1000),
                                          nn.LeakyReLU(0.2),
                                          nn.Linear(1000, 2))

    def forward(self, input):
        return self.discriminator(input)     

class Factor_VAE(nn.Module):

    def __init__(self, nc=1, z_dim=10):
        super(Factor_VAE, self).__init__()

        self.nc = nc 
        self.z_dim = z_dim

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        in_channels = nc
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
        return  [self.decode(z), mu, log_var, z]



