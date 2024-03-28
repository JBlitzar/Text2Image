import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end
        self.block = nn.Sequential(
            nn.Conv2d(start, end, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(end, end, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    def forward(self, x):

        return self.block(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class VAE(nn.Module):
    #https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/slides/lec17.pdf

    #https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
    def __init__(self, device="cpu", encoder=nn.Sequential(), decoder=nn.Sequential(), latent_dim_num=8*8*8, before_latent_dim=16*16*16):

        
        super().__init__()
        self.device = device
        
        self.encoder = encoder
        self.decoder = decoder

        # latent mean and variance 
        self.mean_layer = nn.Linear(before_latent_dim, latent_dim_num)
        self.logvar_layer = nn.Linear(before_latent_dim, latent_dim_num)

        self.expand_layer = nn.Linear(latent_dim_num,before_latent_dim)

    def encode(self, x):
        #slide 22
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)#randn does a normal distribution 
        z = mean + var*epsilon #slide 19
        return z

    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        #slide 22

        # in a class-conditional vae, Y would be appended to the data here becore getting run through this half of the net
        mean, logvar = self.encode(x)

        z = self.reparameterization(mean, logvar)

        z = self.expand_layer(z)
        # in a class-conditional vae, Y would be appended to the data here becore getting run through this half of the net
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    

def COCO_VAE_factory(device="cpu"):
    return VAE(
        device=device, 
        encoder=nn.Sequential(
            ConvBlock(3,256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(256,128),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(128,64),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(64,32),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(32,16),
            
            nn.Flatten()
        ), 
        decoder=nn.Sequential(
            

            Reshape(-1, 16,16,16),

            ConvBlock(16,32),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            ConvBlock(32,64),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            ConvBlock(64,128),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            ConvBlock(128,256),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest
            
            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1 ),
            nn.Sigmoid()

        ), 
        latent_dim_num=8*8*8,
        before_latent_dim=16*16*16
    )


def vae_loss_function(inputs, results, mean, log_var):
    batch_size = inputs.size(0)

    reconstruction_loss = F.binary_cross_entropy(results, inputs, reduction='sum')

    kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    kl_divergence_loss = torch.mean(kl_divergence_loss)

    total_loss = reconstruction_loss + kl_divergence_loss
    
    total_loss /= batch_size
    
    return total_loss, reconstruction_loss, kl_divergence_loss



if __name__ == "__main__":
    with torch.no_grad():
        net = COCO_VAE_factory(device="mps")
        net.eval()
        net.to("mps")
        out, _, _= net(torch.randn(64,3,256,256).to("mps"))
        print(out.size())
        assert tuple(out.size()) == (64,3,256,256)
        print("Done!")