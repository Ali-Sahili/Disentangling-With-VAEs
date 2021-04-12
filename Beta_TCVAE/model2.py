import torch
import torch.nn as nn
from torch.autograd import Variable

import math
from numbers import Number
import Beta_TCVAE.distribution as dist

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

class Beta_TCVAE(nn.Module):
    def __init__(self, args, prior_dist=dist.Normal(), q_dist=dist.Normal()):
        super(Beta_TCVAE, self).__init__()

        self.z_dim = args.z_dim
        self.include_mutinfo = args.include_mutinfo
        self.beta = args.beta_tcvae
        self.mss = args.mss
        self.x_dist = dist.Bernoulli()

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        self.encoder = nn.Sequential(
                          nn.Conv2d(1, 32, 4, 2, 1),  # 32 x 32
                          nn.BatchNorm2d(32),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(32, 32, 4, 2, 1),  # 16 x 16
                          nn.BatchNorm2d(32),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(32, 64, 4, 2, 1),  # 8 x 8
                          nn.BatchNorm2d(64),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(64, 64, 4, 2, 1),  # 4 x 4
                          nn.BatchNorm2d(64),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(64, 512, 4),
                          nn.BatchNorm2d(512),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(512, self.z_dim * self.q_dist.nparams, 1)
                          )


        self.decoder = nn.Sequential(
                          nn.ConvTranspose2d(self.z_dim, 512, 1, 1, 0),  # 1 x 1
                          nn.BatchNorm2d(512),
                          nn.ReLU(inplace=True),
                          nn.ConvTranspose2d(512, 64, 4, 1, 0),  # 4 x 4
                          nn.BatchNorm2d(64),
                          nn.ReLU(inplace=True),
                          nn.ConvTranspose2d(64, 64, 4, 2, 1),  # 8 x 8
                          nn.BatchNorm2d(64),
                          nn.ReLU(inplace=True),
                          nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 x 16
                          nn.BatchNorm2d(32),
                          nn.ReLU(inplace=True),
                          nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 32 x 32
                          nn.BatchNorm2d(32),
                          nn.ReLU(inplace=True),
                          nn.ConvTranspose2d(32, 1, 4, 2, 1)    # 64 x 64
                          )


    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), 1, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        zs = zs.view(-1, self.z_dim, 1, 1)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density( zs.view(batch_size, 1, self.z_dim),
                                   z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams))

        if self.mss:
            # minibatch stratified sampling
            logiw_matrix = Variable(self.log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp( logiw_matrix.view(batch_size, batch_size, 1) + 
                                             _logqz, dim=1, keepdim=False).sum(1)
        else:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - 
                                    math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - 
                                    math.log(batch_size * dataset_size))


        if self.include_mutinfo:
            modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (logqz_prodmarginals - logpz)
        else:
            modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach()
