import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from test import test
from generate_dataset import SimpleDots

from Beta_VAE.train import train
from WAE.train import train_WAE_MMD
from DIP_VAE.train import train_DIP_VAE
from Factor_VAE.train import train_Factor_VAE
from Beta_TCVAE.train import train_beta_TCVAE

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    #data_loader = return_data(args)
    device = 'cuda' if args.use_cuda else 'cpu'
     
    dataset = SimpleDots(image_size=(64, 64), circle_radius=8, num_images=args.nb_imgs)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                               shuffle=True, num_workers=args.num_workers)
    print("Data is generated ...")
    #print("size of training set is: ", len(data_loader.dataset), " images.")
    print("size of training set is: ", len(data_loader)*args.batch_size, " images.")
    print() 
        
    if args.method == "Beta_VAE" or args.method == "Annealed_VAE":
        model, plots = train(args, data_loader, device)
    
    elif args.method == "Factor_VAE":
        model, plots = train_Factor_VAE(args, data_loader, device)

    elif args.method == "Beta_TCVAE":
        model, plots = train_beta_TCVAE(args, data_loader, device)

    elif args.method == "DIP_VAE":
        model, plots = train_DIP_VAE(args, data_loader, device) 
    
    elif args.method == "WAE":
        model, plots = train_WAE_MMD(args, data_loader, device) 
            
    else:
        raise NotImplementedError('Implemented variants are: Beta-VAE, Annealed-VAE, Factor-VAE, Beta-TCVAE and DIP-VAE.')
    

    if args.save_plot: 
        save_plot_path = os.path.join(args.output_dir, "plots/"+args.method)
        if not os.path.exists(save_plot_path):
            os.makedirs(save_plot_path)   

        json_file = json.dumps(plots)
        saving_file = open(save_plot_path + "/plots.json","w")
        saving_file.write(json_file)
        saving_file.close()
        
        nb_iters = len(plots["Recons"])
        
        plt.figure("Reconstruction loss")
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.plot(range(nb_iters), plots["Recons"])
        plt.savefig(save_plot_path + '/recons_loss_' + args.method + '.png')
    
        plt.figure("KLD loss")
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.plot(range(nb_iters), plots["KLD"])
        plt.savefig(save_plot_path + '/kld_loss_' + args.method + '.png')
    
        plt.figure("Total Loss")
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.plot(range(nb_iters), plots["total_loss"])
        plt.savefig(save_plot_path + '/total_loss_' + args.method + '.png')
        plt.show()
        
    test(args, data_loader, model, device, global_iter=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE-Variants for Learning Disentanglement.')
    
    parser.add_argument('--method', default='Beta_VAE', type=str, help='Beta-VAE, Annealed-VAE, Factor-VAE, Beta-TCVAE or DIP-VAE.')

    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--use_cuda', default=True, type=bool, help='enable cuda')
    parser.add_argument('--save_plot', default=True, type=bool, help='save losses plots')
    
    # I/O Paths Parameters
    parser.add_argument('--save_output', default=True, type=bool, help='save images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    
    parser.add_argument('--ckpt_dir', default='checkpoints',type=str, help='ckpoint directory')
    parser.add_argument('--ckpt_iter', default=1000, type=int, help='load specific checkpoint.')

    # Dataset setting Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='toy_dataset', type=str, help='dataset name')
    parser.add_argument('--nb_imgs', default=100, type=int, help='number of input images.')
    parser.add_argument('--image_size', default=64, type=int, help='image size.')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    # training Settings
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--display_step', default=1000, type=int,help='print res every n iters.')
    parser.add_argument('--save_step', default=1000, type=int, help='saving every n iters.')
    
    # Model Parameters
    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the latent space z')
    parser.add_argument('--nc', default=1, type=int, help='number of input channels.')    

    # Beta-VAE Parameters
    parser.add_argument('--beta', default=4, type=float, help='beta parameter beta-VAE')
        
    # Annealed-VAE Parameters
    parser.add_argument('--gamma', default=1000.,type=float, help='gamma parameter Annealed-VAE')
    parser.add_argument('--C_max', default=25.,type=float, help='capacity of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e3,type=float,help='stop increasing capacity')
    
    # Factor-VAE Parameters
    parser.add_argument('--lr_D', default=1e-4, type=float, help='lr of the discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='Adam optim of discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='Adam optim of discriminator')
    parser.add_argument('--fvae_gamma', default=40., type=float, help='gamma of Factor-VAE')
    parser.add_argument('--f_anneal_steps', default=500, type=int,help='annealing of Fact-VAE') 
    
    # Beta-TCVAE Parameters
    parser.add_argument('--btc_alpha', default=1., type=float, help='alpha of beta_TCVAE')
    parser.add_argument('--btc_beta', default=6., type=float, help='beta of beta_TCVAE')
    parser.add_argument('--btc_gamma', default=1., type=float, help='gamma of beta_TCVAE')
    parser.add_argument('--anneal_steps', default=200, type=int, help='gamma of beta_TCVAE')

    # DIP-VAE Parameters
    parser.add_argument('--lambda_diag', default=10., type=float, help='lambda of DIP_VAE')
    parser.add_argument('--lambda_offdiag', default=5., type=float, help='lambda of DIP_VAE')

    # WVAE Parameters
    parser.add_argument('--reg_weight', default=100., type=float, help='lambda of DIP_VAE')
    parser.add_argument('--latent_var', default=2., type=float, help='lambda of DIP_VAE')
    parser.add_argument('--kernel_type', default='imq', type=str, help='kernel type for MMD.')
            
    args = parser.parse_args()

    main(args)
