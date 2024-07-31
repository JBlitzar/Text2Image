import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
#scuffed version of https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[3,8,15,22,29]):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        self.transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        x = self.transforms(x)
        y = self.transforms(y)
        
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += nn.functional.mse_loss(x, y)
        
        return loss