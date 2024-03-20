import torch
import torch.nn as nn
import torch.nn.functional as F
print("VAE architecture loaded.")

class Resize(nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size

    def forward_oneimg(self, x):
        result = F.interpolate(x, size=self.size)
        
        return result
    def forward(self, x):
        return torch.cat([self.forward_oneimg(a) for a in torch.split(x, 1, dim=0)], dim=0)
class Autoencoder(nn.Module):
    def _print(self, item):
        if self.debug:
            print(item)
    def __init__(self, debug=False):
        #compresses 3x640x480 to 3x80x60
        self.debug = debug
        self._print("Running VAE")
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # now its 320x240

            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # now its 160x120

            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # now its 80x60

            # q) how to get to 64x64? a) we dont need to
            Resize(size=(60,60)),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Resize(size=(60,60)), 

            nn.Upsample(scale_factor=2, mode='nearest'),  #now 160x120
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Tanh()
        )
    def forward(self,x):
        self._print(f"Input size: {x.size()}")
        x = self.encoder(x)
        self._print(f"Encoded size: {x.size()}")
        x = self.decoder(x)
        self._print(f"Output size: {x.size()}")
        return x
    
class VAE(nn.Module):
    def _print(self, item):
        if self.debug:
            print(item)
    def __init__(self):
        #compresses 3x640x480 to 3x64x64
        self.debug = False
        self._print("Running VAE")
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # now its 320x240

            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # now its 160x120

            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # now its 80x60


            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU()

        )
        self.decoder = nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  #now 160x120
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Tanh()
        )
    def forward(self,x):
        x = self.encoder(x)
        self._print(f"Encoded size: {x.size()}")
        if self.debug:
            assert list(x.size())[1:]== [3,64,64]
        x = self.decoder(x)
        return x
    
if __name__ == "__main__":
    with torch.no_grad():
        net = VAE(debug=True)
        net.eval()
        net.to("mps")
        net(torch.randn(64,3,640,480).to("mps"))
        print("Done!")