### Implementation of several recent variants of VAE used for disantenglement task.

#### Implemented Variants
* [Beta-VAE](https://openreview.net/pdf?id=Sy2fzU9gl)
* [Annealed-VAE](https://arxiv.org/abs/1804.03599)
* [Beta-TCVAE](https://arxiv.org/abs/1802.04942)
* [Factor-VAE](https://arxiv.org/abs/1802.05983)
* [DIP-VAE](https://arxiv.org/abs/1711.00848) (I & II)
* [WAE](https://arxiv.org/abs/1711.01558di)

#### Requirements
The experiments were performed using Python 3.8.5 with the following Python packages:
- [numpy](http://www.numpy.org/) == 1.18.5
- [tqdm](https://tqdm.github.io/) == 4.30.0
- [torch](https://pytorch.org/) == 1.5.1
- [torchvision](https://pypi.org/project/torchvision/) = 0.6.1
- [imageio](https://pypi.org/project/imageio/) == 2.9.0
- [pillow](https://pillow.readthedocs.io/en/stable/installation.html) == 8.1.0
- [matplotlib](https://pypi.org/project/matplotlib/) == 3.3.3
- [scipy](https://pypi.org/project/scipy/) == 1.5.4
- [sklearn](https://pypi.org/project/scikit-learn/) == 0.24.1

#### Setup
```
python3 main.py --method [METHOD] 
                --dset_dir [DATASET_DIR]
                --use_cuda [USE_CUDA]
                --z_dim [LATENT_DIM]
                --save_output [SAVE_OUTPUT] --output_dir [OUTPUT_DIR]
                --batch_size [BATCH_SIZE]
                --lr [LEARNING_RATE] --num_epochs [NUMBER_OF_EPOCHS]
                --display_step [DISPLAY_STEP]
                --save_step [SAVE_STEP] 

```
#### Acknowledgments
- Implementation of several VAE's variants done by [AntixK](https://github.com/AntixK/PyTorch-VAE).
- Thanks to [davidlmorton](https://github.com/davidlmorton/disentangled) for Visualization part.
