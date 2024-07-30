import torch
import torch.nn as nn
import torch.nn.functional as F

class FunctionalToClass(nn.Module):
    def __init__(self, function, *args, **kwargs) -> None:
        self.function = function
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return self.function(x)

class ShapePrinter(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        print(x.size())
        return x
    
class Thrower(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        raise InterruptedError
        return x
    
class ConditionedSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList()
        for arg in args:
            self.layers.append(arg)
    
    def forward(self, x, t):
        for layer in self.layers:
            x = layer(x,t)
        return x

class AllowExtraArguments(nn.Module):
    def __init__(self, child):
        super().__init__()
        self.child = child
    
    def forward(self, x, *args):
        return self.child(x)
    



class ConvLabelEmbedding(nn.Module):                                                                                                                                     
    def __init__(self, num_classes, embedding_dim, repeat_size):                                                                                                                      
        super(ConvLabelEmbedding, self).__init__()                                                                                                                       
        self.embedding = nn.Embedding(num_classes, embedding_dim) # Linear layer with a few more tricks: Basically a lookup table.
        self.height, self.width = repeat_size                                                                                                
                                                                                                                                                                        
    def forward(self, labels):                                                                                                                            
        # Embed and expand the dimensions to match input spatial dimensions                                                                                              
        label_embeddings = self.embedding(labels)  # (batch_size, embedding_dim)                                                                                         
        label_embeddings = label_embeddings.unsqueeze(-1).unsqueeze(-1)  # (batch_size, embedding_dim, 1, 1)                                                             
        label_embeddings = label_embeddings.expand(-1, -1, self.height, self.width)  # (batch_size, embedding_dim, height, width)                                                  
        return label_embeddings    

    def condition_with_label(self, x, label):
        embedded_label = self.__call__(label)
        return torch.cat([x, embedded_label], dim=1)                                        
                                                                                                                                                                            
class DenseLabelEmbedding(nn.Module):                                                                                                                                    
    def __init__(self, num_classes, embedding_dim):                                                                                                                      
        super(DenseLabelEmbedding, self).__init__()                                                                                                                      
        self.embedding = nn.Embedding(num_classes, embedding_dim)                                                                                                        
                                                                                                                                                                        
    def forward(self, labels):                                                                                                                                                                                                                                                    
        return self.embedding(labels)
    
    def condition_with_label(self, x, label):
        embedded_label = self.__call__(label)
        return x + embedded_label
    



class ConvBlock(nn.Module):
    def __init__(self, start, end=None, activation=nn.ReLU, emb_dim=256):
        super().__init__()
        if end == None:
            end = start * 2

        self.block = nn.Sequential(
            nn.Conv2d(start, end, kernel_size=4, padding=1, stride=2),
            activation(),
        )


        self.activation = activation()

        self.emb_layer = nn.Sequential(

            nn.Linear(
                emb_dim,
                end
            ),
        )

        
    def forward(self, x, label):

        x = self.block(x)
        emb = self.emb_layer(label)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        
        return x + emb
class ConvTransposeBlock(nn.Module):
    def __init__(self, start, end, activation=nn.ReLU, emb_dim=256):
        super().__init__()
        self.activation = activation()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(start, end, kernel_size=4, padding=1, stride=2),
            activation(),

        )

        self.emb_layer = nn.Sequential(

            nn.Linear(
                emb_dim,
                end
            ),
        )
        
    def forward(self, x, label):
        x = self.block(x)
        emb = self.emb_layer(label)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        
        return x + emb
    
class VAE(nn.Module):
    def __init__(self, encoder, decoder, hdim, device="cpu", before_latent=None):
        super(VAE, self).__init__()

        if before_latent == None:
            before_latent = hdim

        self.device = device

        # Define encoder layers
        self.encoder = encoder

        self.mean_layer = nn.Linear(before_latent, hdim)

        self.var_layer = nn.Linear(before_latent, hdim)

        # Define decoder layers
        self.decoder = decoder
    
    def encode(self, x):
        x = self.encoder(x)

        mean = self.mean_layer(x)

        var = self.var_layer(x)

        z = self.reperameterize(mean, var)

        return z, mean, var
    
    def reperameterize(self, mean, variance):

        std = torch.exp(variance/2)

        epsilon = torch.randn_like(variance).to(self.device)

        return mean+std*epsilon


    def forward(self, x):
        # Define forward pass
        z, mean, var = self.encode(x)


        x = self.decoder(z)
        return x, mean, var

class CVAE(VAE):
    def __init__(self, encoder, decoder, hdim, num_classes, input_size, device="cpu", before_latent=None, embedding_channels=1):
        super().__init__(encoder, decoder, hdim, device, before_latent)

        self.num_classes = num_classes
        self.input_size = input_size


    

    def encode(self, x, label):
        

        x = self.encoder(x, label)

        mean = self.mean_layer(x)

        var = self.var_layer(x)

        z = self.reperameterize(mean, var)

        

        return z, mean, var
    
    def decode(self, z, label):
        

        

        z = self.decoder(z, label)
        return z
    
    def forward(self, x, label):
        # Define forward pass
        z, mean, var = self.encode(x, label)


        x = self.decode(z, label)
        return x, mean, var

    


    

def VAE_loss(x, reconstruction, mean, variance, kl_weight=1):
    #https://github.com/pytorch/examples/blob/main/vae/main.py
    def print_(string):
        for i in range(10):
            print(string)

    if torch.isnan(x).any() or torch.isnan(reconstruction).any():
        print_("NaNs detected in reconstruction or x")
    

    BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    if torch.isnan(BCE).any():
        print_("NaNs detected in BCE")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if torch.isnan(variance).any() or torch.isnan(mean).any():
        print_("NaNs detected in variance or mean")
    KLD = -0.5 * torch.sum(1 + variance - mean.pow(2) - variance.exp())
    if torch.isnan(KLD).any():
        print_("NaNs detected in KLD")
    return BCE + KLD * kl_weight, BCE, KLD * kl_weight


def COCO_CVAE_factory(device="cpu", start_depth=64, num_classes=768):
    hidden_dimension = start_depth*4*8*8
    before_latent=start_depth*4*8*8
    return CVAE(
        encoder=ConditionedSequential(


            #64^2, 3
            ConvBlock(3, start_depth, emb_dim=num_classes),
            ConvBlock(start_depth, start_depth * 2, emb_dim=num_classes),
            ConvBlock(start_depth * 2, start_depth * 4, emb_dim=num_classes), 


            AllowExtraArguments(nn.Flatten()),
            AllowExtraArguments(nn.Linear(start_depth*4*8*8, start_depth*4*8*8)),
            AllowExtraArguments(nn.ReLU()),

            
        ),
        decoder=ConditionedSequential(
            AllowExtraArguments(nn.Linear(start_depth*4*8*8, start_depth*4*8*8)),
            AllowExtraArguments(nn.ReLU()),
            AllowExtraArguments(nn.Unflatten(1,(start_depth*4,8,8))),

            ConvTransposeBlock(start_depth*4, start_depth*2, emb_dim=num_classes),
            ConvTransposeBlock(start_depth*2, start_depth, emb_dim=num_classes),
            ConvTransposeBlock(start_depth, 3,  emb_dim=num_classes),
            
            




        ),
        hdim=hidden_dimension,
        num_classes=num_classes,
        input_size=(3,64,64),
        before_latent=before_latent,
        device=device,
        embedding_channels=num_classes
    ), hidden_dimension

if __name__ == "__main__":
    test_data = torch.randn((5,3,64,64)).to("mps")
    test_label = torch.randn((5,768)).to("mps")

    model, _ = COCO_CVAE_factory(device="mps")
    model.to("mps")

    result = model(test_data, test_label)