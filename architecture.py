import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict

class Unet(nn.Module):
    def __init__(self):
        self.quiet = True
        super(Unet, self).__init__()

        #https://www.desmos.com/calculator/jbopjvrrmj
        # based off of celebaunet class

        # Encoder layers
        self.encoder = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)), #3x640x480 -> 64x640x480
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)), # 64x640x482 -> 64x640x482
    ('relu2', nn.ReLU()),
    ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)), # 64x640x484 -> 32x640x484 

    ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
    ('relu3', nn.ReLU()),
    ('conv4', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
    ('relu4', nn.ReLU()),
    ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),

    ('conv5', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
    ('relu5', nn.ReLU()),
    ('conv6', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
    ('relu6', nn.ReLU()),
    ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),
]))

        # Decoder layers
        self.decoder = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
    ('relu1', nn.ReLU()),
    ('upsample1', nn.Upsample(scale_factor=2, mode='nearest')),

    ('conv2', nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
    ('relu2', nn.ReLU()),
    ('conv3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
    ('relu3', nn.ReLU()),
    ('upsample2', nn.Upsample(scale_factor=2, mode='nearest')),

    ('conv4', nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1)),
    ('relu4', nn.ReLU()),
    ('conv5', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
    ('relu5', nn.ReLU()),
    ('upsample3', nn.Upsample(scale_factor=2, mode='nearest')),
    ('conv6', nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
    ("relu6",nn.ReLU()),
    ('conv7', nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)),
    ('sigmoid', nn.Sigmoid())
]))
    def tile_prompt(self, prompt_tensor, desired_shape): # by chatgpt, edited by me
        
        num_channels = desired_shape[1]
        
        dividend = (num_channels*desired_shape[0]*desired_shape[2]*desired_shape[3])
        divisor = prompt_tensor.size(0)

        repeat_times = dividend // divisor
        

        tiled_tensor = prompt_tensor.repeat(repeat_times,1)
        remainder = dividend % divisor
        tiled_tensor = torch.cat((tiled_tensor, tiled_tensor[:,:remainder]), dim=1)
        

        tiled_tensor = tiled_tensor.view(*desired_shape)
        
        assert tuple(tiled_tensor.size()) == desired_shape

        return tiled_tensor
    def forward(self, x, prompt):
        #prompt = torch.cat((prompt, prompt[:496]))


        # Define forward pass
        if not self.quiet:
            print("Input Size:", x.size())

        skip_connections = []

        # Encoder forward pass with saving skip connections
        for layer_name, layer in self.encoder.named_children():
            
            if "pool" in layer.__class__.__name__.lower():
                
                
                x = layer(x)
                x = torch.cat((x, self.tile_prompt(prompt, tuple(x.size()))), dim=1)
                skip_connections.append(x.clone())  # Save skip connection, which includes the tiled prompt
            else: 
                x = layer(x)
            if not self.quiet:
                print(f"{layer.__class__.__name__} Output Size:", x.size())
           
        if not self.quiet:
            print("\n=====Begin decoding=====\n")
        skipnum = 1
        # Decoder forward pass with using skip connections
        for layer_name, layer in self.decoder.named_children():
            if "Upsample" in layer.__class__.__name__:
                if(layer_name != "upsample3"):
                    x = torch.cat((x, skip_connections[-(skipnum)]), dim=0)  # Concatenate with skip connection
                    x = layer(x)
                    
                    skipnum += 1
                else:
                    x = layer(x)
            else:
                x = layer(x)

            if not self.quiet:
                print(f"{layer.__class__.__name__} Output Size:", x.size())

        return x

if __name__ == "__main__":
    with torch.no_grad():
        myUnet = Unet()
        myUnet.eval()
        myUnet(torch.randn(64,3,640,480), torch.randn(64,784))