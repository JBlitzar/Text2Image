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

class ShapePrinter(nn.Module):
    def __init__(self, prefix=""):
        super().__init__()
        self.prefix = prefix
    
    def forward(self,x):
        print(self.prefix + str(x.size()))
        return x

class DenseBlock(nn.Module):
    def __init__(self, start, end, residual=False):
        super().__init__()
        self.start = start
        self.end = end
        self.block = nn.Sequential(
            nn.Linear(start, end),
            nn.ReLU(),
            nn.Linear(end, end),
            nn.ReLU()
        )
        self.residual = residual
    def forward(self, x):
        if self.residual:
            return F.relu(x + self.block(x))
        else:
            return self.block(x)



class DenseChain(nn.Module):
    def __init__(self, size, length, residual=False):
        super().__init__()
        modules = []
        for _ in range(length):
            modules.append(DenseBlock(size, size, residual))
        self.block = nn.Sequential(*modules)
        
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
            ConvBlock(3,128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # ConvBlock(256,128),

            # nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(128,64),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(64,32),

            # nn.MaxPool2d(kernel_size=2, stride=2),

            # ConvBlock(32,16),
            
            nn.Flatten()
        ), 
        decoder=nn.Sequential(
            
            Reshape(-1, 32,32,32),
            #Reshape(-1, 16,16,16),

            # ConvBlock(16,32),
            # nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            ConvBlock(32,64),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            ConvBlock(64,128),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            # ConvBlock(128,256),
            
            # nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest
            
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1 ),
            nn.Sigmoid()

        ), 
        latent_dim_num=16*16*16,#8*8*8,
        before_latent_dim=32*32*32
    )

def COCO_T2I_VAE_factory(device="cpu"):
    return VAE(
        device=device, 
        encoder=nn.Sequential(
            nn.Flatten(),
            DenseBlock(3840, 7680, False),
            DenseBlock(7680, 15360, False),
            DenseBlock(15360, 30720, False),
            DenseBlock(30720, 32768, False),
        ), 
        decoder=nn.Sequential(
            #ShapePrinter("decoder "),
            
            Reshape(-1, 32,32,32),
            #ShapePrinter("reshaped "),
            #Reshape(-1, 16,16,16),

            # ConvBlock(16,32),
            # nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            ConvBlock(32,64),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            #ShapePrinter("halfway-decoded "),

            ConvBlock(64,128),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            # ConvBlock(128,256),
            
            # nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest
            
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1 ),
            nn.Sigmoid(),
            #ShapePrinter("output ")

        ), 
        latent_dim_num=16*16*16,#8*8*8,
        before_latent_dim=32*32*32
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
        # net = COCO_VAE_factory(device="mps")
        # net.eval()
        # net.to("mps")
        # out, _, _= net(torch.randn(64,3,256,256).to("mps"))
        # print(out.size())
        # assert tuple(out.size()) == (64,3,256,256)
        # print("Done!")
        net = COCO_T2I_VAE_factory(device="mps")
        net.eval()
        net.to("mps")
        out, _, _= net(torch.randn(64,768).to("mps"))
        print(out.size())
        assert tuple(out.size()) == (64,3,128,128)
        print("Done!")