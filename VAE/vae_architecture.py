import torch
import torch.nn as nn
import torch.nn.functional as F
print("VAE architecture loaded.")
#test
class Resize(nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode='nearest')
class VAE(nn.Module):
    def _print(self, item):
        if self.debug:
            print(item)
    def __init__(self, debug=False):
        #compresses 3x640x480 to 3x64x64
        self.debug = debug
        self._print("Running VAE")
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # now its 320x240

            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # now its 160x120

            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # now its 80x60

            # how to get to 64x64?
            Resize(size=(64,64)),
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

        )
        self.decoder = nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            Resize(size=(60,80)), 

            nn.Upsample(scale_factor=2, mode='nearest'),  #now 160x120
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU()
        )
    def forward(self,x):
        _print = self._print
        _print(x.size())
        x = self.encoder(x)
        _print(f"Encoded size: {x.size()}")
        if self.debug:
            assert list(x.size())[1:]== [3,64,64]
        x = self.decoder(x)
        _print(x.size())
        return x
    



if __name__ == "__main__":
    with torch.no_grad():
        net = VAE(debug=True)
        net.eval()
        net.to("mps")
        net(torch.randn(64,3,480,640).to("mps"))
        print("Done!")